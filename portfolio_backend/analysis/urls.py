from django.urls import path

from .views import (
    ScreenerResultsView, RunScanView, TickerDetailView, TickerChartView,
    LabPositionMarkManualView, LabPerformanceView, RejectionAuditView,
    ManualTickerCheckView,
)

urlpatterns = [
    path("screener/<slug:preset_slug>/results/", ScreenerResultsView.as_view(), name="analysis-screener-results"),
    path("screener/<slug:preset_slug>/run/", RunScanView.as_view(), name="analysis-screener-run"),
    path("ticker/<str:ticker>/", TickerDetailView.as_view(), name="analysis-ticker-detail"),
    path("ticker/<str:ticker>/chart/", TickerChartView.as_view(), name="analysis-ticker-chart"),
    path("ticker/<str:ticker>/manual-check/", ManualTickerCheckView.as_view(), name="analysis-ticker-manual-check"),
    path("lab/<int:position_id>/mark-manual/", LabPositionMarkManualView.as_view(), name="analysis-lab-mark-manual"),
    path("lab/performance/", LabPerformanceView.as_view(), name="analysis-lab-performance"),
    path("rejections/audit/", RejectionAuditView.as_view(), name="analysis-rejections-audit"),
]
