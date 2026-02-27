import os
import requests
from dotenv import load_dotenv


def send_test_signal() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, 'portfolio_backend', '.env')
    load_dotenv(env_path)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    text = (
        "🧪 **TEST DU SYSTÈME D'APPROBATION**\n\n"
        "🎯 **Ticker :** TD.TO (Toronto)\n"
        "💰 **Prix Actuel :** 82.45 CAD\n"
        "📊 **Confiance IA :** 78.4%\n"
        "🛡️ **Stop ATR (2.0x) :** 79.10 CAD\n\n"
        "⚠️ *Ceci est une simulation. Cliquer sur 'Approuver' testera ta logique d'achat sans placer d'ordre réel.*"
    )

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "✅ APPROUVER (Test)", "callback_data": "approve_test_TD.TO"},
                    {"text": "❌ REJETER", "callback_data": "reject_test_TD.TO"},
                ]
            ]
        },
    }

    response = requests.post(url, json=payload, timeout=10)
    if response.status_code == 200:
        print("✅ Signal de test envoyé avec succès ! Vérifie ton Telegram.")
    else:
        print(f"❌ Erreur lors de l'envoi : {response.text}")


if __name__ == "__main__":
    send_test_signal()
