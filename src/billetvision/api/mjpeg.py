"""MJPEG video streaming utilities."""
import time
from typing import Generator
from fastapi.responses import StreamingResponse

def frame_generator() -> Generator[bytes, None, None]:
    """Generates MJPEG boundary frames."""
    # Placeholder frame generator until pipeline feeds live frames
    while True:
        time.sleep(0.1)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
