"""Telegram notification dispatcher (optional, enabled by env vars / config)."""
import os
import logging
from pathlib import Path
from typing import Optional
import requests

logger = logging.getLogger(__name__)

def send_telegram_alert(message: str, image_path: Optional[str | Path] = None) -> bool:
    """Send alert via Telegram Bot API if configured."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return False

    try:
        if image_path and Path(image_path).exists():
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as photo:
                resp = requests.post(url, data={"chat_id": chat_id, "caption": message}, files={"photo": photo}, timeout=5)
                return resp.status_code == 200
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
