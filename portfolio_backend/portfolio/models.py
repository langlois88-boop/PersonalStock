from django.conf import settings
from django.db import models


class Account(models.Model):
	ACCOUNT_TYPES = (
		('TFSA', 'TFSA'),
		('CRI', 'CRI'),
		('CASH', 'CASH'),
	)

	user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
	name = models.CharField(max_length=50)
	account_type = models.CharField(max_length=10, choices=ACCOUNT_TYPES)

	def __str__(self) -> str:
		return f"{self.name} ({self.account_type})"


class Stock(models.Model):
	symbol = models.CharField(max_length=10)
	name = models.CharField(max_length=100)
	sector = models.CharField(max_length=50)
	target_weight = models.FloatField()
	dividend_yield = models.FloatField()
	latest_price = models.FloatField(null=True, blank=True)
	latest_price_updated_at = models.DateTimeField(null=True, blank=True)
	day_low = models.FloatField(null=True, blank=True)
	day_high = models.FloatField(null=True, blank=True)

	def __str__(self) -> str:
		return f"{self.symbol}"


class AccountTransaction(models.Model):
	TRANSACTION_TYPES = (
		('BUY', 'BUY'),
		('SELL', 'SELL'),
		('DIVIDEND', 'DIVIDEND'),
	)

	account = models.ForeignKey(Account, on_delete=models.CASCADE)
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	date = models.DateField()
	type = models.CharField(max_length=10, choices=TRANSACTION_TYPES)
	quantity = models.FloatField()
	price = models.FloatField()

	def __str__(self) -> str:
		return f"{self.account.name} {self.type} {self.quantity} {self.stock.symbol}"


class Dividend(models.Model):
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	date = models.DateField()
	amount = models.FloatField()

	def __str__(self) -> str:
		return f"{self.stock.symbol} {self.date}"


class Prediction(models.Model):
	RECOMMENDATIONS = (
		('BUY', 'BUY'),
		('HOLD', 'HOLD'),
		('SELL', 'SELL'),
	)

	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	date = models.DateField()
	predicted_price = models.FloatField()
	recommendation = models.CharField(max_length=10, choices=RECOMMENDATIONS)

	def __str__(self) -> str:
		return f"{self.stock.symbol} {self.date} {self.recommendation}"


class PaperTrade(models.Model):
	STATUS_CHOICES = (
		('OPEN', 'OPEN'),
		('CLOSED', 'CLOSED'),
	)

	ticker = models.CharField(max_length=10)
	entry_price = models.DecimalField(max_digits=10, decimal_places=2)
	quantity = models.IntegerField()
	entry_date = models.DateTimeField(auto_now_add=True)
	entry_signal = models.FloatField(null=True, blank=True)
	entry_features = models.JSONField(null=True, blank=True)
	stop_loss = models.DecimalField(max_digits=10, decimal_places=2)
	status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='OPEN')
	pnl = models.DecimalField(max_digits=10, decimal_places=2, default=0)
	exit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
	exit_date = models.DateTimeField(null=True, blank=True)
	notes = models.TextField(blank=True, default='')

	class Meta:
		ordering = ['-entry_date']

	def __str__(self) -> str:
		return f"{self.ticker} {self.status}"


class Portfolio(models.Model):
	name = models.CharField(max_length=50)
	capital = models.FloatField()
	created_at = models.DateTimeField(auto_now_add=True)
	stocks = models.ManyToManyField(Stock, through='Transaction')

	def __str__(self) -> str:
		return self.name


class Transaction(models.Model):
	portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	shares = models.FloatField()
	price_per_share = models.FloatField()
	date = models.DateField()
	transaction_type = models.CharField(
		max_length=10,
		choices=(('BUY', 'BUY'), ('SELL', 'SELL')),
	)

	def __str__(self) -> str:
		return f"{self.transaction_type} {self.shares} {self.stock.symbol}"


class PortfolioHolding(models.Model):
	portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	shares = models.FloatField()
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		unique_together = ('portfolio', 'stock')

	def __str__(self) -> str:
		return f"{self.portfolio.name} {self.stock.symbol}: {self.shares}"


class NewsArticle(models.Model):
	symbol = models.CharField(max_length=10)
	title = models.CharField(max_length=300)
	description = models.TextField(blank=True)
	url = models.URLField(unique=True)
	source = models.CharField(max_length=100, blank=True)
	published_at = models.DateTimeField(null=True, blank=True)
	fetched_at = models.DateTimeField(auto_now_add=True)
	raw = models.JSONField(null=True, blank=True)

	def __str__(self) -> str:
		return f"{self.symbol}: {self.title}"


class StockNews(models.Model):
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	headline = models.CharField(max_length=300)
	url = models.URLField(unique=True)
	published_at = models.DateTimeField(null=True, blank=True)
	sentiment = models.FloatField(null=True, blank=True)
	source = models.CharField(max_length=100, blank=True)
	raw = models.JSONField(null=True, blank=True)
	fetched_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-published_at']

	def __str__(self) -> str:
		return f"{self.stock.symbol}: {self.headline}"


class PriceHistory(models.Model):
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	date = models.DateField()
	close_price = models.FloatField()

	class Meta:
		unique_together = ('stock', 'date')
		ordering = ['-date']

	def __str__(self) -> str:
		return f"{self.stock.symbol} {self.date}"


class DividendHistory(models.Model):
	stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
	date = models.DateField()
	dividend_yield = models.FloatField()

	class Meta:
		unique_together = ('stock', 'date')
		ordering = ['-date']

	def __str__(self) -> str:
		return f"{self.stock.symbol} {self.date}"


class AlertEvent(models.Model):
	category = models.CharField(max_length=50)
	message = models.TextField()
	stock = models.ForeignKey(Stock, on_delete=models.SET_NULL, null=True, blank=True)
	portfolio = models.ForeignKey(Portfolio, on_delete=models.SET_NULL, null=True, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-created_at']

	def __str__(self) -> str:
		return f"{self.category}: {self.created_at}"


class DripSnapshot(models.Model):
	portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
	as_of = models.DateField()
	capital = models.FloatField()
	dividend_income = models.FloatField()
	yield_rate = models.FloatField()

	class Meta:
		unique_together = ('portfolio', 'as_of')
		ordering = ['-as_of']

	def __str__(self) -> str:
		return f"{self.portfolio.name} {self.as_of}"


class PortfolioDigest(models.Model):
	portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
	period_start = models.DateField()
	period_end = models.DateField()
	summary = models.TextField()
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-created_at']

	def __str__(self) -> str:
		return f"{self.portfolio.name} {self.period_end}"


class UserPreference(models.Model):
	user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
	risk_score = models.FloatField(default=0.5)
	preferred_sectors = models.JSONField(default=dict, blank=True)
	last_feedback = models.TextField(blank=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self) -> str:
		return f"{self.user_id} prefs"


class MacroIndicator(models.Model):
	date = models.DateField(unique=True)
	sp500_close = models.FloatField()
	vix_index = models.FloatField()
	interest_rate_10y = models.FloatField()
	inflation_rate = models.FloatField()
	oil_price = models.FloatField(null=True, blank=True)

	class Meta:
		ordering = ['-date']

	def __str__(self) -> str:
		return f"Macro {self.date}"


class PennyStockUniverse(models.Model):
	symbol = models.CharField(max_length=20, unique=True)
	name = models.CharField(max_length=200, blank=True)
	exchange = models.CharField(max_length=50, blank=True)
	sector = models.CharField(max_length=100, blank=True)
	industry = models.CharField(max_length=120, blank=True)
	price = models.FloatField(null=True, blank=True)
	market_cap = models.FloatField(null=True, blank=True)
	volume = models.FloatField(null=True, blank=True)
	updated_at = models.DateTimeField(auto_now=True)
	raw = models.JSONField(default=dict, blank=True)

	def __str__(self) -> str:
		return f"{self.symbol}"


class PennyStockSnapshot(models.Model):
	stock = models.ForeignKey(PennyStockUniverse, on_delete=models.CASCADE)
	as_of = models.DateField()
	price = models.FloatField(null=True, blank=True)
	market_cap = models.FloatField(null=True, blank=True)
	volume = models.FloatField(null=True, blank=True)
	shares_outstanding = models.FloatField(null=True, blank=True)
	revenue = models.FloatField(null=True, blank=True)
	debt = models.FloatField(null=True, blank=True)
	cash = models.FloatField(null=True, blank=True)
	burn_rate = models.FloatField(null=True, blank=True)
	rsi = models.FloatField(null=True, blank=True)
	macd_hist = models.FloatField(null=True, blank=True)
	sentiment_score = models.FloatField(null=True, blank=True)
	social_mentions = models.IntegerField(null=True, blank=True)
	dilution_score = models.FloatField(null=True, blank=True)
	ai_score = models.FloatField(null=True, blank=True)
	flags = models.JSONField(default=dict, blank=True)
	raw = models.JSONField(default=dict, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		unique_together = ('stock', 'as_of')
		ordering = ['-as_of', '-ai_score']

	def __str__(self) -> str:
		return f"{self.stock.symbol} {self.as_of}"


class PennySignal(models.Model):
	symbol = models.CharField(max_length=10)
	as_of = models.DateField()
	pattern_score = models.FloatField()
	sentiment_score = models.FloatField()
	hype_score = models.FloatField()
	liquidity_score = models.FloatField()
	combined_score = models.FloatField()
	last_price = models.FloatField(null=True, blank=True)
	avg_volume = models.FloatField(null=True, blank=True)
	mentions = models.IntegerField(default=0)
	data = models.JSONField(default=dict, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		unique_together = ('symbol', 'as_of')
		ordering = ['-combined_score']

	def __str__(self) -> str:
		return f"{self.symbol} {self.as_of}"
