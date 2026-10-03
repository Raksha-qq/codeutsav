"""Alert manager: manages alert state, debouncing, and dispatching to web/audio/telegram."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class Alert:
    timestamp: str
    billet_id: str
    status: str
    reasons: List[str]
    image_path: Optional[str] = None

class AlertManager:
    """Manages active and historical alerts."""

    def __init__(self):
        self.history: List[Alert] = []
        self.active_alert: Optional[Alert] = None

    def trigger(self, billet_id: str, status: str, reasons: List[str], image_path: Optional[str] = None) -> Alert:
        alert = Alert(
            timestamp=datetime.utcnow().isoformat(),
            billet_id=billet_id,
            status=status,
            reasons=reasons,
            image_path=image_path
        )
        self.history.append(alert)
        self.active_alert = alert
        return alert

    def clear_active(self) -> None:
        self.active_alert = None
