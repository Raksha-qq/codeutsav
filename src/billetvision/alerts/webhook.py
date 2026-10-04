"""Generic webhook notifier (optional, enabled by URL in env var or config)."""
import logging
import os
from typing import Callable, Optional

import requests

from billetvision.alerts.manager import Alert
from billetvision.alerts.telegram import send_telegram_alert

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "BILLETVISION_WEBHOOK_URL"


def send_webhook(alert: Alert, url: Optional[str] = None) -> bool:
    """POST ``alert`` as JSON to the configured webhook URL.

    Returns False without any network I/O when no URL is configured.
    """
    target = url or os.environ.get(WEBHOOK_ENV)
    if not target:
        return False
    try:
        resp = requests.post(target, json=alert.to_payload(), timeout=5)
        return 200 <= resp.status_code < 300
    except Exception as exc:
        logger.error("Webhook delivery failed: %s", exc)
        return False


def telegram_notifier(alert: Alert) -> None:
    """Notifier adapter: Telegram message (+ snapshot) for FAIL/REWORK/REVIEW/system alerts."""
    send_telegram_alert(alert.message(), alert.image_path)


def build_notifiers(alerts_cfg: dict) -> "list[Callable[[Alert], None]]":
    """Build the notifier list from the ``alerts`` config section.

    Telegram self-gates on TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars.  The
    webhook is added only when ``webhook_url`` (or the env var) is non-empty.
    """
    notifiers: "list[Callable[[Alert], None]]" = [telegram_notifier]
    url = alerts_cfg.get("webhook_url") or os.environ.get(WEBHOOK_ENV)
    if url:

        def _webhook(alert: Alert, _url: str = url) -> None:
            send_webhook(alert, _url)

        notifiers.append(_webhook)
    return notifiers
