from django.db import transaction as db_transaction
from rest_framework import serializers

from .models import (
    Account,
    AccountTransaction,
    AlertEvent,
    Dividend,
    DripSnapshot,
    Portfolio,
    PortfolioDigest,
    PortfolioHolding,
    PriceHistory,
    Prediction,
    PennyStockSnapshot,
    PaperTrade,
    PennyStockUniverse,
    PennySignal,
    Stock,
    StockNews,
    Transaction,
    UserPreference,
    MacroIndicator,
)


class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = '__all__'


class TransactionSerializer(serializers.ModelSerializer):
    stock_details = StockSerializer(source='stock', read_only=True)

    class Meta:
        model = Transaction
        fields = '__all__'

    def validate(self, attrs):
        portfolio = attrs.get('portfolio')
        stock = attrs.get('stock')
        shares = float(attrs.get('shares') or 0)
        price = float(attrs.get('price_per_share') or 0)
        tx_type = attrs.get('transaction_type')

        if shares <= 0 or price <= 0:
            raise serializers.ValidationError('Shares and price_per_share must be positive.')

        if tx_type == 'BUY':
            cost = shares * price
            if portfolio.capital < cost:
                raise serializers.ValidationError('Not enough capital for BUY.')
        elif tx_type == 'SELL':
            holding = PortfolioHolding.objects.filter(portfolio=portfolio, stock=stock).first()
            if not holding or holding.shares < shares:
                raise serializers.ValidationError('Not enough shares to SELL.')
        else:
            raise serializers.ValidationError('transaction_type must be BUY or SELL.')

        return attrs

    def create(self, validated_data):
        portfolio = validated_data['portfolio']
        stock = validated_data['stock']
        shares = float(validated_data['shares'])
        price = float(validated_data['price_per_share'])
        tx_type = validated_data['transaction_type']

        with db_transaction.atomic():
            tx = super().create(validated_data)

            holding, _ = PortfolioHolding.objects.get_or_create(
                portfolio=portfolio,
                stock=stock,
                defaults={'shares': 0},
            )

            if tx_type == 'BUY':
                portfolio.capital = float(portfolio.capital or 0) - (shares * price)
                holding.shares = float(holding.shares or 0) + shares
            else:
                portfolio.capital = float(portfolio.capital or 0) + (shares * price)
                holding.shares = float(holding.shares or 0) - shares

            portfolio.save(update_fields=['capital'])

            if holding.shares <= 0:
                holding.delete()
            else:
                holding.save(update_fields=['shares'])

        return tx


class PortfolioSerializer(serializers.ModelSerializer):
    stocks = StockSerializer(many=True, read_only=True)
    transactions = TransactionSerializer(many=True, read_only=True, source='transaction_set')
    holdings = serializers.SerializerMethodField()

    class Meta:
        model = Portfolio
        fields = '__all__'

    def get_holdings(self, obj):
        holdings = PortfolioHolding.objects.select_related('stock').filter(portfolio=obj)
        return [
            {
                'stock': h.stock.symbol,
                'stock_name': h.stock.name,
                'shares': h.shares,
            }
            for h in holdings
        ]


class PortfolioHoldingSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    stock_name = serializers.CharField(source='stock.name', read_only=True)

    class Meta:
        model = PortfolioHolding
        fields = '__all__'


class PriceHistorySerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)

    class Meta:
        model = PriceHistory
        fields = '__all__'


class StockNewsSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    summary = serializers.SerializerMethodField()

    class Meta:
        model = StockNews
        fields = '__all__'

    def get_summary(self, obj):
        raw = obj.raw or {}
        return raw.get('summary') or raw.get('description') or ''


class DripSnapshotSerializer(serializers.ModelSerializer):
    portfolio_name = serializers.CharField(source='portfolio.name', read_only=True)

    class Meta:
        model = DripSnapshot
        fields = '__all__'


class AlertEventSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    portfolio_name = serializers.CharField(source='portfolio.name', read_only=True)

    class Meta:
        model = AlertEvent
        fields = '__all__'


class AccountSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = Account
        fields = '__all__'


class AccountTransactionSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    account_name = serializers.CharField(source='account.name', read_only=True)

    class Meta:
        model = AccountTransaction
        fields = '__all__'


class DividendSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)

    class Meta:
        model = Dividend
        fields = '__all__'


class PredictionSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    stock_latest_price = serializers.SerializerMethodField()
    reason = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = '__all__'

    def get_stock_latest_price(self, obj):
        if obj.stock and obj.stock.latest_price is not None:
            return float(obj.stock.latest_price)
        last = PriceHistory.objects.filter(stock=obj.stock).order_by('-date').first()
        return float(last.close_price) if last else None

    def get_reason(self, obj):
        last_price = self.get_stock_latest_price(obj)
        predicted = float(obj.predicted_price or 0)
        if not last_price or last_price == 0:
            return 'Insufficient recent price data.'
        delta = (predicted - last_price) / last_price
        pct = abs(delta) * 100
        if delta > 0.02:
            return f'Forecast {pct:.2f}% above current price.'
        if delta < -0.02:
            return f'Forecast {pct:.2f}% below current price.'
        return 'Forecast close to current price.'


class PaperTradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperTrade
        fields = '__all__'


class PortfolioDigestSerializer(serializers.ModelSerializer):
    portfolio_name = serializers.CharField(source='portfolio.name', read_only=True)

    class Meta:
        model = PortfolioDigest
        fields = '__all__'


class UserPreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserPreference
        fields = '__all__'


class MacroIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = MacroIndicator
        fields = '__all__'


class PennySignalSerializer(serializers.ModelSerializer):
    class Meta:
        model = PennySignal
        fields = '__all__'


class PennyStockUniverseSerializer(serializers.ModelSerializer):
    class Meta:
        model = PennyStockUniverse
        fields = '__all__'


class PennyStockSnapshotSerializer(serializers.ModelSerializer):
    stock_symbol = serializers.CharField(source='stock.symbol', read_only=True)
    stock_name = serializers.CharField(source='stock.name', read_only=True)
    stock_exchange = serializers.CharField(source='stock.exchange', read_only=True)

    class Meta:
        model = PennyStockSnapshot
        fields = '__all__'


class PennyStockScoutSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol', read_only=True)
    company_name = serializers.CharField(source='name', read_only=True)
    latest_price = serializers.SerializerMethodField()
    latest_sentiment = serializers.SerializerMethodField()
    ai_score = serializers.SerializerMethodField()

    class Meta:
        model = PennyStockUniverse
        fields = [
            'ticker',
            'company_name',
            'sector',
            'exchange',
            'latest_price',
            'latest_sentiment',
            'ai_score',
        ]

    def _get_latest_snapshot(self, obj):
        prefetched = getattr(obj, 'latest_snapshots', None)
        if isinstance(prefetched, list) and prefetched:
            return prefetched[0]
        return PennyStockSnapshot.objects.filter(stock=obj).order_by('-as_of').first()

    def get_latest_price(self, obj):
        snapshot = self._get_latest_snapshot(obj)
        return float(snapshot.price) if snapshot and snapshot.price is not None else None

    def get_latest_sentiment(self, obj):
        snapshot = self._get_latest_snapshot(obj)
        if not snapshot:
            return None
        score = snapshot.sentiment_score
        label = 'Neutral'
        if score is not None:
            if score >= 0.2:
                label = 'Positive'
            elif score <= -0.2:
                label = 'Negative'
        return {
            'score': float(score) if score is not None else None,
            'buzz': snapshot.social_mentions,
            'label': label,
        }

    def get_ai_score(self, obj):
        snapshot = self._get_latest_snapshot(obj)
        if not snapshot:
            return {'score': 0, 'confidence': 0, 'trend': 'N/A'}
        flags = snapshot.flags or {}
        confidence = (
            float(flags.get('fundamental_score', 0))
            + float(flags.get('technical_score', 0))
            + float(flags.get('sentiment_norm', 0))
            + float(flags.get('liquidity_score', 0))
        ) / 4
        trend = 'Neutral'
        if snapshot.rsi is not None and snapshot.rsi < 30 and (snapshot.macd_hist or 0) > 0:
            trend = 'Reversal'
        elif snapshot.ai_score is not None and snapshot.ai_score >= 70:
            trend = 'Accumulation'
        elif snapshot.ai_score is not None and snapshot.ai_score <= 30:
            trend = 'Distribution'
        return {
            'score': float(snapshot.ai_score or 0),
            'confidence': round(confidence, 2),
            'trend': trend,
        }
