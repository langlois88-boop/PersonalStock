import math
import statistics
import openai
import joblib
from datetime import datetime


class DanasEngine:
	def __init__(self) -> None:
		self.client = openai.OpenAI(
			base_url="http://100.88.73.110:8001/v1",
			api_key="danas-local-key",
		)
		self.model_name = "deepseek-r1"
		self.models = {
			'penny': joblib.load('/mnt/BackupSSD/ai_data/models/penny_model.pkl'),
			'bluechip': joblib.load('/mnt/BackupSSD/ai_data/models/bluechip_model.pkl'),
		}

	def get_ml_prediction(self, ticker_data, category: str = 'bluechip') -> dict:
		model = self.models.get(category)
		prediction = model.predict(ticker_data)
		probability = model.predict_proba(ticker_data)
		return {
			"signal": "BUY" if prediction[0] == 1 else "SELL",
			"confidence": round(max(probability[0]) * 100, 2),
		}

	def ask_danas(self, user_query: str, context_data=None):
		now = datetime.now().strftime("%d/%m/%Y %H:%M")
		system_prompt = (
			f"Tu es Danas, l'IA de trading. Nous sommes le {now}. "
			f"Contexte technique : {context_data if context_data else 'Pas de data live'}. "
			"Réponds en français. Analyse le Sharpe Ratio et le Broker Lag (-0.58s) si nécessaire."
		)
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_query},
			],
			stream=True,
		)
		return response

	def backtest_stress_test(self, trade_returns: list[float], fee_pct: float = 0.002) -> dict:
		def _normalize_return(value: float) -> float:
			try:
				val = float(value)
			except (TypeError, ValueError):
				return 0.0
			if abs(val) > 1:
				val /= 100.0
			return val

		if not trade_returns:
			return {
				'adjusted_returns': [],
				'sharpe': None,
				'sharpe_above_1': False,
			}

		adjusted = [
			(_normalize_return(value) - fee_pct)
			for value in trade_returns
		]
		mean_ret = statistics.mean(adjusted) if adjusted else 0.0
		stdev_ret = statistics.pstdev(adjusted) if len(adjusted) > 1 else 0.0
		sharpe = None
		if stdev_ret > 0:
			sharpe = (mean_ret / stdev_ret) * math.sqrt(len(adjusted))
		return {
			'adjusted_returns': adjusted,
			'sharpe': sharpe,
			'sharpe_above_1': bool(sharpe is not None and sharpe > 1.0),
		}
