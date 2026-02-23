from django.urls import include, path
from rest_framework import routers

from .views import (
    AccountTransactionViewSet,
    AccountViewSet,
    AlertEventViewSet,
    AnalystInsightsView,
    AIRecommendationsRunView,
    AIOpportunityView,
    AIScoutView,
    AIScoutBatchView,
    ScoutFundamentalsView,
    BacktestView,
    AIBacktesterView,
    PortfolioOptimizerView,
    SectorTrendCompareView,
    ForecastView,
    DripSnapshotViewSet,
    DividendViewSet,
    PortfolioHoldingViewSet,
    PortfolioViewSet,
    PortfolioDigestViewSet,
    PriceHistoryViewSet,
    PredictionViewSet,
    PaperTradeViewSet,
    PennySignalViewSet,
    StockNewsViewSet,
    StockViewSet,
    TransactionViewSet,
    UserPreferenceViewSet,
    PaperTradeSummaryView,
    PaperTradePerformanceView,
    PaperTradeEquityCurveView,
    PaperTradeExplanationLogView,
    PortfolioCsvImportView,
    StockSearchView,
    PennyStockAnalyticsView,
    PennyStockScoutView,
    PennyStockPredictionView,
    StablePredictionView,
    BluechipHunterView,
    MacroIndicatorView,
    AlpacaIntradayView,
    MarketScannerView,
    AccountDashboardView,
    PortfolioDashboardView,
    PortfolioNewsView,
    SentimentScannerView,
    HealthCheckView,
    DataQADailyView,
    ModelRegistryView,
    ModelCalibrationDailyView,
    ModelDriftDailyView,
    ModelEvaluationDailyView,
    ModelMonitoringSummaryView,
    SandboxWatchlistView,
)

router = routers.DefaultRouter()
router.register(r'stocks', StockViewSet)
router.register(r'portfolios', PortfolioViewSet)
router.register(r'transactions', TransactionViewSet)
router.register(r'prices', PriceHistoryViewSet)
router.register(r'holdings', PortfolioHoldingViewSet)
router.register(r'stocknews', StockNewsViewSet)
router.register(r'drip-snapshots', DripSnapshotViewSet)
router.register(r'alerts', AlertEventViewSet)
router.register(r'accounts', AccountViewSet)
router.register(r'account-transactions', AccountTransactionViewSet)
router.register(r'dividends', DividendViewSet)
router.register(r'predictions', PredictionViewSet)
router.register(r'paper-trades', PaperTradeViewSet)
router.register(r'digests', PortfolioDigestViewSet)
router.register(r'preferences', UserPreferenceViewSet)
router.register(r'penny-signals', PennySignalViewSet)

urlpatterns = [
    path('stocks/search/', StockSearchView.as_view()),
    path('penny-analytics/', PennyStockAnalyticsView.as_view()),
    path('penny-scout/', PennyStockScoutView.as_view()),
    path('penny-predict/<str:symbol>/', PennyStockPredictionView.as_view()),
    path('bluechip-hunter/', BluechipHunterView.as_view()),
    path('macro/', MacroIndicatorView.as_view()),
    path('alpaca/intraday/', AlpacaIntradayView.as_view()),
    path('market/scanner/', MarketScannerView.as_view()),
    path('paper-trades/summary/', PaperTradeSummaryView.as_view()),
    path('paper-trades/performance/', PaperTradePerformanceView.as_view()),
    path('paper-trades/equity/', PaperTradeEquityCurveView.as_view()),
    path('paper-trades/explanations/', PaperTradeExplanationLogView.as_view()),
    path('watchlist/', SandboxWatchlistView.as_view()),
    path('dashboard/portfolio/', PortfolioDashboardView.as_view()),
    path('dashboard/news/', PortfolioNewsView.as_view()),
    path('dashboard/sentiment/', SentimentScannerView.as_view()),
    path('dashboard/accounts/', AccountDashboardView.as_view()),
    path('predict/stable/<str:symbol>/', StablePredictionView.as_view()),
    path('import/portfolio/', PortfolioCsvImportView.as_view()),
    path('qa/daily/', DataQADailyView.as_view()),
    path('models/evaluation/', ModelEvaluationDailyView.as_view()),
    path('models/calibration/', ModelCalibrationDailyView.as_view()),
    path('models/drift/', ModelDriftDailyView.as_view()),
    path('models/monitoring/', ModelMonitoringSummaryView.as_view()),
    path('models/registry/', ModelRegistryView.as_view()),
    path('analyst/', AnalystInsightsView.as_view()),
    path('ai/recommendations/run/', AIRecommendationsRunView.as_view()),
    path('ai/opportunities/', AIOpportunityView.as_view()),
    path('ai/scout/', AIScoutView.as_view()),
    path('ai/scout/batch/', AIScoutBatchView.as_view()),
    path('scout/fundamentals/', ScoutFundamentalsView.as_view()),
    path('backtest/', BacktestView.as_view()),
    path('backtester/', AIBacktesterView.as_view()),
    path('optimizer/', PortfolioOptimizerView.as_view()),
    path('sector-compare/', SectorTrendCompareView.as_view()),
    path('forecast/', ForecastView.as_view()),
    path('health/', HealthCheckView.as_view()),
    path('', include(router.urls)),
]
