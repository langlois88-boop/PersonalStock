from django.db import models


class ScreenerPreset(models.Model):
    """
    Un preset de screener (ex. "Undervalued Without Reason"). Le champ
    `thresholds` surcharge les seuils par défaut de
    `analysis.services.quant_filter.SECTOR_THRESHOLDS` (voir merge_thresholds).

    Générique par design : "Value + Catalyst" (module SEC EDGAR insider
    buying) et d'autres presets futurs se branchent ici sans migration
    supplémentaire — seulement une nouvelle ligne + les seuils/flags voulus.
    """

    slug = models.SlugField(max_length=80, unique=True)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, default='')
    thresholds = models.JSONField(default=dict, blank=True)
    # False pour un preset purement quantitatif (aucun appel LLM, étapes 2/3 sautées).
    requires_news_check = models.BooleanField(default=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class ScanRun(models.Model):
    STATUS_CHOICES = (
        ('running', 'running'),
        ('completed', 'completed'),
        ('failed', 'failed'),
    )

    preset = models.ForeignKey(ScreenerPreset, on_delete=models.CASCADE, related_name='runs')
    universe_source = models.CharField(max_length=20, default='sandboxes')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='running', db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    tickers_scanned = models.IntegerField(default=0)
    candidates_after_quant_filter = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-started_at']

    def __str__(self) -> str:
        return f"{self.preset.slug} @ {self.started_at:%Y-%m-%d %H:%M} ({self.status})"


class ScanResult(models.Model):
    VERDICT_CHOICES = (
        ('confirmed', 'confirmed'),
        ('uncertain', 'uncertain'),
        ('rejected_deepseek', 'rejected_deepseek'),
        ('rejected_claude', 'rejected_claude'),
        # Garde-fou de sécurité (2026-08-14, cas HAIN) : bloqué par
        # analysis.services.sanity_check APRÈS le verdict LLM (peu importe
        # s'il disait confirmed/uncertain) -- distinct d'un rejet LLM, voir
        # SanityCheckResult et anomaly_reason ci-dessous. claude_verdict
        # garde le verdict LLM d'origine pour traçabilité complète.
        ('flagged_anomaly', 'flagged_anomaly'),
    )

    scan_run = models.ForeignKey(ScanRun, on_delete=models.CASCADE, related_name='results')
    ticker = models.CharField(max_length=10, db_index=True)
    sector = models.CharField(max_length=50, blank=True, default='')
    quant_score = models.FloatField(default=0.0)
    quant_details = models.JSONField(default=dict, blank=True)
    price_at_scan = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)

    deepseek_verdict = models.CharField(max_length=20, blank=True, default='')
    deepseek_reasoning = models.TextField(blank=True, default='')

    claude_verdict = models.CharField(max_length=20, blank=True, default='')
    claude_reasoning = models.TextField(blank=True, default='')
    claude_tokens_used = models.IntegerField(default=0)
    # verify_ticker() (claude_verifier.py) retourne déjà 'low'/'medium'/'high'
    # dans ClaudeResult.confidence mais ne le stockait nulle part avant le
    # 2026-08-14 (audit ML_LAB_FUTURE_MODEL.md, Partie 2) -- utilisé puis
    # perdu. Feature potentielle pour un futur modèle entraîné sur les
    # résultats réels (verdict LLM + confiance -> performance).
    claude_confidence = models.CharField(max_length=10, blank=True, default='')

    # Raison précise du garde-fou de sécurité (analysis.services.sanity_check)
    # quand final_verdict == 'flagged_anomaly' -- vide sinon. Distinct de
    # claude_reasoning/deepseek_reasoning : ce texte vient d'un calcul de
    # seuils déterministe, pas d'un LLM.
    anomaly_reason = models.TextField(blank=True, default='')

    VERDICT_SOURCE_CHOICES = (
        # Un seul appel Claude (CLAUDE_VOTE_CALLS<=1) -- comportement
        # d'origine, avant le vote majoritaire du 2026-08-09.
        ('single_call', 'single_call'),
        # Plusieurs appels, verdicts convergents -- claude_verdict reflète
        # le consensus.
        ('llm_consensus', 'llm_consensus'),
        # Plusieurs appels, verdicts divergents -- claude_verdict forcé à
        # 'uncertain', distinct d'un uncertain qui viendrait d'un manque
        # d'info (voir claude_verifier.py::verify_ticker).
        ('llm_disagreement', 'llm_disagreement'),
    )
    verdict_source = models.CharField(max_length=20, choices=VERDICT_SOURCE_CHOICES, blank=True, default='single_call')

    final_verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES, db_index=True)

    # Réservé à l'intégration future du module de patterns de chandeliers
    # (portfolio/patterns.py) comme signal de timing sur les tickers déjà
    # confirmés — pas encore alimenté dans ce lot (voir brief, "Prochaines
    # étapes").
    candle_pattern_detected = models.CharField(max_length=40, blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['ticker', 'created_at'], name='analysis_scanres_tkr_dt'),
        ]

    def __str__(self) -> str:
        return f"{self.ticker} {self.final_verdict} ({self.scan_run_id})"


class FundamentalLabPosition(models.Model):
    """
    Un pick du screener qui entre dans le sandbox FUNDAMENTAL_LAB — un seul
    sandbox pour confirmed et uncertain (visuellement distingués côté
    frontend via final_verdict). `manually_selected` marque les picks
    qu'Eric choisit lui-même parmi les résultats, sans créer de position
    dupliquée : permet de comparer screener complet vs picks manuels vs
    benchmark.
    """

    scan_result = models.OneToOneField(ScanResult, on_delete=models.CASCADE, related_name='lab_position')
    ticker = models.CharField(max_length=10, db_index=True)
    preset = models.ForeignKey(ScreenerPreset, on_delete=models.CASCADE, related_name='lab_positions')

    entry_price = models.DecimalField(max_digits=12, decimal_places=4)
    entry_date = models.DateTimeField()
    benchmark_price_at_entry = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)

    manually_selected = models.BooleanField(default=False)
    manual_note = models.TextField(blank=True, default='')
    is_open = models.BooleanField(default=True)

    CLOSE_REASON_CHOICES = (
        ('stop_loss', 'stop_loss'),
        ('manual', 'manual'),
        # Ajoutés le 2026-08-23 : avant ça, monitor_lab_positions n'avait
        # qu'un stop-loss -- une position gagnante ne se fermait JAMAIS
        # d'elle-même, seuls les perdants se fermaient. Impossible de
        # mesurer honnêtement la performance du screener avec un
        # échantillon de positions fermées structurellement biaisé vers
        # les pertes. Voir monitor_lab_positions pour les seuils.
        ('take_profit', 'take_profit'),
        ('max_hold_time', 'max_hold_time'),
    )
    # Ajoutés le 2026-08-14 (audit ML_LAB_FUTURE_MODEL.md, Partie 2) : avant
    # ça, une position fermée n'était traçable que via son paper_trade lié
    # (PaperTrade.exit_price/exit_date, et la raison de fermeture seulement
    # en texte libre dans PaperTrade.notes, pas machine-lisible). Dupliqués
    # ici plutôt que de forcer une jointure à chaque requête ML future --
    # cette table est le futur feature/label set, elle doit être
    # auto-suffisante. Renseignés au moment de la fermeture (voir
    # analysis/tasks.py::monitor_lab_positions pour stop_loss).
    exit_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    exit_date = models.DateTimeField(null=True, blank=True)
    close_reason = models.CharField(max_length=20, choices=CLOSE_REASON_CHOICES, blank=True, default='')
    realized_return_pct = models.FloatField(null=True, blank=True)

    # Ordre paper trade réel (portfolio.PaperTrade, sandbox='FUNDAMENTAL_LAB').
    # Nullable : si la création de l'ordre échoue, la position lab existe
    # quand même (traçabilité du pick) mais sans ordre — l'échec est loggé
    # explicitement, jamais silencieux (cf. incident "zéro trade").
    paper_trade = models.ForeignKey(
        'portfolio.PaperTrade', on_delete=models.SET_NULL, null=True, blank=True, related_name='+',
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-entry_date']

    def __str__(self) -> str:
        return f"{self.ticker} lab position ({self.entry_date:%Y-%m-%d})"


class RejectionLog(models.Model):
    REJECTED_BY_CHOICES = (
        ('deepseek', 'deepseek'),
        ('claude', 'claude'),
    )

    scan_result = models.OneToOneField(ScanResult, on_delete=models.CASCADE, related_name='rejection_log')
    ticker = models.CharField(max_length=10, db_index=True)
    preset = models.ForeignKey(ScreenerPreset, on_delete=models.CASCADE, related_name='rejection_logs')

    rejected_by = models.CharField(max_length=10, choices=REJECTED_BY_CHOICES, db_index=True)
    reasoning_text = models.TextField(blank=True, default='')
    quant_score = models.FloatField(default=0.0)

    price_at_rejection = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    # Prix du benchmark (^GSPTSE) au moment du rejet — nécessaire pour calculer
    # un alpha réel dans check_rejection_outcomes (voir TODO #4 du brief).
    benchmark_price_at_rejection = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)

    date_rejection = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-date_rejection']

    def __str__(self) -> str:
        return f"{self.ticker} rejected by {self.rejected_by} ({self.date_rejection:%Y-%m-%d})"


class RejectionOutcome(models.Model):
    """
    Suivi périodique (30/60/90j) d'un RejectionLog : rendement réalisé
    depuis le rejet vs le benchmark. `possible_missed_winner` flag les cas
    où le filtre (surtout DeepSeek) a peut-être laissé passer un gagnant.
    """

    rejection_log = models.ForeignKey(RejectionLog, on_delete=models.CASCADE, related_name='outcomes')
    days_since_rejection = models.IntegerField()

    price_then = models.FloatField()
    price_now = models.FloatField()
    return_pct = models.FloatField()
    benchmark_return_pct = models.FloatField()
    alpha = models.FloatField(db_index=True)
    possible_missed_winner = models.BooleanField(default=False, db_index=True)

    checked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-checked_at']
        constraints = [
            models.UniqueConstraint(
                fields=['rejection_log', 'days_since_rejection'], name='analysis_one_outcome_per_window',
            ),
        ]

    def __str__(self) -> str:
        return f"{self.rejection_log.ticker} +{self.days_since_rejection}j alpha={self.alpha}"


class CustomWatchlistTicker(models.Model):
    """
    Survivants du balayage large hebdomadaire (S&P 500 + TSX Composite,
    seuils quantitatifs stricts -- voir analysis/services/broad_scan_filter.py)
    -- alimente l'univers quotidien du screener (`_get_universe`,
    universe_source='combined') une fois qu'un premier cycle a produit des
    résultats sensés. `consecutive_weeks_missed` implémente une sortie
    progressive plutôt qu'un retrait immédiat : un titre qui ne repasse pas
    les seuils une semaine donnée n'est pas forcément un faux positif
    ponctuel (données manquantes, volatilité passagère d'un ratio) --
    is_active ne bascule à False qu'après 3 semaines manquées d'affilée.
    """

    SOURCE_CHOICES = (
        ('SP500', 'SP500'),
        ('TSX', 'TSX'),
    )

    ticker = models.CharField(max_length=20, unique=True, db_index=True)
    source_index = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_confirmed = models.DateTimeField(auto_now=True)
    consecutive_weeks_missed = models.IntegerField(default=0)
    piotroski_score = models.IntegerField(null=True, blank=True)
    altman_z_score = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)

    class Meta:
        ordering = ['-last_confirmed']

    def __str__(self) -> str:
        return f"{self.ticker} ({self.source_index}) active={self.is_active}"


class ManualTickerCheck(models.Model):
    """
    Vérification manuelle à la demande (2026-08-13), sur n'importe quel
    ticker tapé depuis la fiche ticker -- complètement isolée du pipeline
    automatique. Réutilise les mêmes fonctions (evaluate_ticker,
    triage_ticker, verify_ticker) mais ne crée JAMAIS de ScanResult, de
    FundamentalLabPosition, de CustomWatchlistTicker ni de PaperTrade.
    Cette table existe uniquement pour ne pas perdre un raisonnement
    Claude déjà payé -- ce n'est pas un signal d'achat, pas un scan, pas
    une entrée dans un sandbox. Voir analysis/services/manual_check.py.
    """

    ticker = models.CharField(max_length=20, db_index=True)
    checked_at = models.DateTimeField(auto_now_add=True, db_index=True)

    sector = models.CharField(max_length=50, blank=True, default='')
    quant_score = models.FloatField(default=0.0)
    quant_details = models.JSONField(default=dict, blank=True)
    price_at_check = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)

    deepseek_verdict = models.CharField(max_length=20, blank=True, default='')
    deepseek_reasoning = models.TextField(blank=True, default='')

    claude_verdict = models.CharField(max_length=20, blank=True, default='')
    claude_reasoning = models.TextField(blank=True, default='')
    claude_tokens_used = models.IntegerField(default=0)

    final_verdict = models.CharField(max_length=20, blank=True, default='')

    class Meta:
        ordering = ['-checked_at']

    def __str__(self) -> str:
        return f"{self.ticker} manual check {self.final_verdict} ({self.checked_at:%Y-%m-%d %H:%M})"


class LabWeeklySuggestion(models.Model):
    """
    Résumé hebdomadaire automatique (2026-08-14, Partie 3 -- voir
    analysis.tasks.generate_lab_weekly_suggestions). Cherche des patterns
    simples et objectifs (corrélation entre un facteur -- ROE bas, FCF Yield
    extrême, secteur, confiance Claude -- et la performance réelle des
    positions FUNDAMENTAL_LAB fermées dans la semaine). Ne modifie JAMAIS
    un seuil ou un comportement automatiquement -- une suggestion texte
    courte, jamais appliquée sans confirmation explicite d'Eric. Affiché
    dans un encart sur la page Analytics & ML Lab existante (pas une
    nouvelle page séparée).
    """

    week_start = models.DateField(db_index=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    positions_closed_analyzed = models.IntegerField(default=0)
    has_pattern = models.BooleanField(default=False)
    summary = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-generated_at']

    def __str__(self) -> str:
        tag = "pattern" if self.has_pattern else "aucun pattern"
        return f"Suggestion semaine du {self.week_start} ({tag})"
