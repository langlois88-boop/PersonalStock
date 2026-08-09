from django.urls import path

from .views import (
    ScreenerResultsView, RunScanView, TickerDetailView,
    LabPositionMarkManualView, LabPerformanceView, RejectionAuditView,
)

urlpatterns = [
    path("screener/<slug:preset_slug>/results/", ScreenerResultsView.as_view(), name="analysis-screener-results"),
    path("screener/<slug:preset_slug>/run/", RunScanView.as_view(), name="analysis-screener-run"),
    path("ticker/<str:ticker>/", TickerDetailView.as_view(), name="analysis-ticker-detail"),
    path("lab/<int:position_id>/mark-manual/", LabPositionMarkManualView.as_view(), name="analysis-lab-mark-manual"),
    path("lab/performance/", LabPerformanceView.as_view(), name="analysis-lab-performance"),
    path("rejections/audit/", RejectionAuditView.as_view(), name="analysis-rejections-audit"),
]
