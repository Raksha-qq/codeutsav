"""Tests for end-to-end BilletVision inspection pipeline lifecycle."""

import pytest
from billetvision.pipeline import BilletVisionPipeline


def test_pipeline_lifecycle():
    pipeline = BilletVisionPipeline(config_path="config/config.yaml")
    assert pipeline.is_running is False

    pipeline.start()
    assert pipeline.is_running is True

    pipeline.stop()
    assert pipeline.is_running is False
