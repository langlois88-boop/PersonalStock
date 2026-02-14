from django.contrib import admin

from .models import (
	Account,
	AccountTransaction,
	AlertEvent,
	Dividend,
	DividendHistory,
	DripSnapshot,
	NewsArticle,
	Portfolio,
	PortfolioHolding,
	PortfolioDigest,
	Prediction,
	PaperTrade,
	Stock,
	StockNews,
	Transaction,
	UserPreference,
	PennySignal,
)


admin.site.register(Stock)
admin.site.register(Portfolio)
admin.site.register(Transaction)
admin.site.register(NewsArticle)
admin.site.register(StockNews)
admin.site.register(DripSnapshot)
admin.site.register(PortfolioHolding)
admin.site.register(DividendHistory)
admin.site.register(AlertEvent)
admin.site.register(Account)
admin.site.register(AccountTransaction)
admin.site.register(Dividend)
admin.site.register(Prediction)
admin.site.register(PaperTrade)
admin.site.register(PortfolioDigest)
admin.site.register(UserPreference)
admin.site.register(PennySignal)

# Register your models here.
