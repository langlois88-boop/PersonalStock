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
    BacktestView,
    AIBacktesterView,
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
    PortfolioCsvImportView,
    StockSearchView,
    PennyStockAnalyticsView,
    PennyStockScoutView,
    PennyStockPredictionView,
    StablePredictionView,
    BluechipHunterView,
    MacroIndicatorView,
    PortfolioDashboardView,
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
    path('paper-trades/summary/', PaperTradeSummaryView.as_view()),
    path('dashboard/portfolio/', PortfolioDashboardView.as_view()),
    path('predict/stable/<str:symbol>/', StablePredictionView.as_view()),
    path('import/portfolio/', PortfolioCsvImportView.as_view()),
    path('analyst/', AnalystInsightsView.as_view()),
    path('ai/recommendations/run/', AIRecommendationsRunView.as_view()),
    path('ai/opportunities/', AIOpportunityView.as_view()),
    path('ai/scout/', AIScoutView.as_view()),
    path('ai/scout/batch/', AIScoutBatchView.as_view()),
    path('backtest/', BacktestView.as_view()),
    path('backtester/', AIBacktesterView.as_view()),
    path('sector-compare/', SectorTrendCompareView.as_view()),
    path('forecast/', ForecastView.as_view()),
    path('', include(router.urls)),
]
