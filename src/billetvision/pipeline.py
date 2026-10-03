"""Pipeline coordinating capture, vision processing, OCR, decision, logging, and events."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BilletVisionPipeline:
    """End-to-end BilletVision inspection pipeline."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.is_running = False

    def start(self):
        """Start capture and processing threads."""
        self.is_running = True
        logger.info("BilletVision pipeline started.")

    def stop(self):
        """Stop capture and processing threads."""
        self.is_running = False
        logger.info("BilletVision pipeline stopped.")
