"""
server/app.py
Required by OpenEnv multi-mode deployment validator.
Imports the FastAPI app from api/main.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

__all__ = ["app"]


def serve():
    """Entry point for uv run serve."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host    = "0.0.0.0",
        port    = 7860,
        workers = 1,
    )
    