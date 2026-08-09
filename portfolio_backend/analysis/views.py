"""
Endpoints DRF pour le module Analyse. Enregistrés via analysis/urls.py,
inclus depuis portfolio_backend/urls.py sous 'api/analysis/'.
"""

from django.db.models import Avg, Count
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    ScreenerPreset, ScanRun, ScanResult, FundamentalLabPosition,
    RejectionLog, RejectionOutcome,
)
from .tasks import run_daily_scan


# --- Serializers ---

class ScanResultSerializer(serializers.ModelSerializer):
    # AnalysisScreener.jsx en a besoin pour le bouton "Je crois en celle-là "
    # (handleMarkManual) : id de la FundamentalLabPosition liée, pas celui
    # du ScanResult. Absent si le résultat a été rejeté (pas de lab_position).
    lab_position_id = serializers.SerializerMethodField()

    class Meta:
        model = ScanResult
        fields = [
            "id", "ticker", "sector", "quant_score", "quant_details",
            "price_at_scan", "deepseek_verdict", "deepseek_reasoning",
            "claude_verdict", "claude_reasoning", "final_verdict",
            "candle_pattern_detected", "created_at", "lab_position_id",
        ]

    def get_lab_position_id(self, obj):
        lab_position = getattr(obj, "lab_position", None)
        return lab_position.id if lab_position else None


class LabPositionSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField()
    final_verdict = serializers.CharField(source="scan_result.final_verdict")
    current_return_pct = serializers.SerializerMethodField()

    class Meta:
        model = FundamentalLabPosition
        fields = [
            "id", "ticker", "final_verdict", "entry_price", "entry_date",
            "manually_selected", "manual_note", "is_open",
            "current_return_pct",
        ]

    def get_current_return_pct(self, obj):
        current_price = self.context.get("current_prices", {}).get(obj.ticker)
        if current_price and obj.entry_price:
            return round((float(current_price) - float(obj.entry_price)) / float(obj.entry_price) * 100, 2)
        return None


# --- Vues ---

class ScreenerResultsView(APIView):
    """GET : derniers résultats de screener pour un preset donné."""

    def get(self, request, preset_slug):
        preset = ScreenerPreset.objects.filter(slug=preset_slug).first()
        if not preset:
            return Response({"error": "Preset introuvable."}, status=status.HTTP_404_NOT_FOUND)

        latest_run = ScanRun.objects.filter(preset=preset, status="completed").order_by("-started_at").first()
        if not latest_run:
            return Response({"results": [], "message": "Aucun scan complété pour ce preset."})

        results = latest_run.results.exclude(
            final_verdict__in=["rejected_deepseek", "rejected_claude"]
        ).select_related("lab_position").order_by("-quant_score")

        return Response({
            "scan_run_id": latest_run.id,
            "scanned_at": latest_run.finished_at,
            "tickers_scanned": latest_run.tickers_scanned,
            "candidates_found": latest_run.candidates_after_quant_filter,
            "results": ScanResultSerializer(results, many=True).data,
        })


class RunScanView(APIView):
    """
    POST : déclenche un scan à la demande (bouton "Lancer l'analyse").
    Lance en tâche async — le frontend doit poller ScreenerResultsView
    plutôt qu'attendre la réponse HTTP.
    """

    def post(self, request, preset_slug):
        preset = ScreenerPreset.objects.filter(slug=preset_slug, is_active=True).first()
        if not preset:
            return Response({"error": "Preset introuvable ou inactif."}, status=status.HTTP_404_NOT_FOUND)

        task = run_daily_scan.delay(preset_slug=preset_slug)
        return Response({"task_id": task.id, "status": "started"}, status=status.HTTP_202_ACCEPTED)


class TickerDetailView(APIView):
    """
    GET : fiche individuelle pour un ticker — combine le dernier
    ScanResult connu (s'il existe) avec les données fraîches.
    """

    def get(self, request, ticker):
        from .services.quant_filter import evaluate_ticker

        ticker = ticker.upper()
        latest_result = ScanResult.objects.filter(ticker=ticker).order_by("-created_at").first()

        # Évalue en direct (pas de cache de fraîcheur pour l'instant — un
        # seul appel yfinance/market_data par consultation de fiche).
        fresh_quant = evaluate_ticker(ticker, preset_thresholds={})

        history = list(
            ScanResult.objects.filter(ticker=ticker).order_by("created_at")
            .values("created_at", "quant_score", "final_verdict")[:60]
        )

        return Response({
            "ticker": ticker,
            "current_data": fresh_quant.details if fresh_quant else None,
            "latest_verdict": latest_result.final_verdict if latest_result else None,
            "latest_reasoning": {
                "claude": latest_result.claude_reasoning if latest_result else None,
                "deepseek": latest_result.deepseek_reasoning if latest_result else None,
            },
            "score_history": list(history),
        })


class LabPositionMarkManualView(APIView):
    """POST : flag une position comme 'Je crois en celle-là ' (manually_selected=True)."""

    def post(self, request, position_id):
        position = FundamentalLabPosition.objects.filter(id=position_id).first()
        if not position:
            return Response({"error": "Position introuvable."}, status=status.HTTP_404_NOT_FOUND)

        position.manually_selected = True
        position.manual_note = request.data.get("note", "")
        position.save(update_fields=["manually_selected", "manual_note"])
        return Response({"status": "ok"})


class LabPerformanceView(APIView):
    """
    GET : dashboard de comparaison — screener complet vs picks manuels,
    par verdict (confirmed/uncertain), avec vs benchmark.
    """

    def get(self, request):
        from .services.quant_filter import fetch_ticker_data

        positions = FundamentalLabPosition.objects.select_related("scan_result").all()
        current_prices = {}
        for ticker in positions.values_list("ticker", flat=True).distinct():
            data = fetch_ticker_data(ticker)
            if data:
                current_prices[ticker] = data["price"]

        serialized = LabPositionSerializer(
            positions, many=True, context={"current_prices": current_prices}
        ).data

        def avg_return(items):
            vals = [i["current_return_pct"] for i in items if i["current_return_pct"] is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        confirmed = [p for p in serialized if p["final_verdict"] == "confirmed"]
        uncertain = [p for p in serialized if p["final_verdict"] == "uncertain"]
        manual = [p for p in serialized if p["manually_selected"]]

        return Response({
            "confirmed": {"count": len(confirmed), "avg_return_pct": avg_return(confirmed)},
            "uncertain": {"count": len(uncertain), "avg_return_pct": avg_return(uncertain)},
            "manual_picks": {"count": len(manual), "avg_return_pct": avg_return(manual)},
            "positions": serialized,
        })


class RejectionAuditView(APIView):
    """
    GET : dashboard d'audit des rejets — DeepSeek vs Claude, avec
    les 'occasions manquées' (alpha > seuil) mises en évidence.
    """

    def get(self, request):
        outcomes = RejectionOutcome.objects.select_related("rejection_log")

        summary = {}
        for source in ("deepseek", "claude"):
            source_outcomes = outcomes.filter(rejection_log__rejected_by=source)
            agg = source_outcomes.aggregate(avg_alpha=Avg("alpha"), count=Count("id"))
            missed = source_outcomes.filter(possible_missed_winner=True).count()
            summary[source] = {
                "total_outcomes_tracked": agg["count"] or 0,
                "avg_alpha_pct": round(agg["avg_alpha"], 2) if agg["avg_alpha"] else None,
                "possible_missed_winners": missed,
            }

        flagged = outcomes.filter(possible_missed_winner=True).select_related("rejection_log").order_by("-alpha")[:20]
        flagged_data = [
            {
                "ticker": o.rejection_log.ticker,
                "rejected_by": o.rejection_log.rejected_by,
                "reasoning": o.rejection_log.reasoning_text,
                "days_since_rejection": o.days_since_rejection,
                "alpha": o.alpha,
            }
            for o in flagged
        ]

        return Response({"summary": summary, "flagged_for_review": flagged_data})
