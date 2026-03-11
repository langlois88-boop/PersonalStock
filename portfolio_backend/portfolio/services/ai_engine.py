import math
import os
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as rt
import openai


class DanasEngine:
    def __init__(self) -> None:
        base_url = (
            os.getenv("DANAS_BASE_URL")
            or os.getenv("OLLAMA_CHAT_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).strip().rstrip("/")
        if "/v1" not in base_url:
            base_url = f"{base_url}/v1"
        api_key = os.getenv("DANAS_API_KEY", "danas-local-key")
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = os.getenv("DANAS_MODEL", os.getenv("OLLAMA_MODEL", "deepseek-r1"))
        self.sessions = self._load_onnx_sessions()

    def _load_onnx_sessions(self) -> dict[str, rt.InferenceSession]:
        registry_dir = Path(os.getenv("DANAS_ONNX_REGISTRY_DIR", "/app/portfolio/ml_engine/models/registry"))
        stable_path = Path(os.getenv("DANAS_ONNX_STABLE_PATH", ""))
        penny_path = Path(os.getenv("DANAS_ONNX_PENNY_PATH", ""))
        fallback_stable = Path(os.getenv("STABLE_ONNX_PATH", "/app/portfolio/ml_engine/models/stable_brain_v1.onnx"))
        fallback_penny = Path(os.getenv("PENNY_ONNX_PATH", "/app/portfolio/ml_engine/models/scout_brain_v1.onnx"))

        def resolve_model_path(model_key: str, override: Path, fallback: Path) -> Path:
            if override and override.exists():
                return override
            latest = registry_dir / model_key / "latest" / "model.onnx"
            if latest.exists():
                return latest
            if registry_dir.exists():
                candidates = sorted((registry_dir / model_key).glob("v*/model.onnx"))
                if candidates:
                    return candidates[-1]
            return fallback

        sessions: dict[str, rt.InferenceSession] = {}
        stable_file = resolve_model_path("stable", stable_path, fallback_stable)
        penny_file = resolve_model_path("penny", penny_path, fallback_penny)
        for key, path in (("bluechip", stable_file), ("penny", penny_file)):
            if path and path.exists():
                sessions[key] = rt.InferenceSession(str(path))
        return sessions

    def _predict_with_session(self, session: rt.InferenceSession, ticker_data) -> tuple[int, float]:
        vector = np.asarray(ticker_data, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: vector})
        probability = None
        for output in outputs:
            if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[-1] >= 2:
                probability = float(output[0][1])
                break
        if probability is None and outputs:
            try:
                probability = float(outputs[0][0])
            except Exception:
                probability = 0.0
        label = 1 if probability >= 0.5 else 0
        return label, probability

    def get_ml_prediction(self, ticker_data, category: str = 'bluechip') -> dict:
        model = self.sessions.get(category)
        if model is None:
            return {"signal": "SELL", "confidence": 0.0}
        prediction, probability = self._predict_with_session(model, ticker_data)
        return {
            "signal": "BUY" if prediction == 1 else "SELL",
            "confidence": round(float(probability) * 100, 2),
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
