"""
server/app.py
OpenEnv multi-mode deployment entry point.
Validator requires: main() function + if __name__ == '__main__' guard.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app   # re-export the FastAPI app

__all__ = ["app", "main"]


def main():
    """
    Server entry point.
    Called by: uv run serve
    Called by: openenv validate (multi-mode deployment check)
    """
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host    = "0.0.0.0",
        port    = int(os.environ.get("PORT", 7860)),
        workers = 1,
        reload  = False,
    )


if __name__ == "__main__":
    main()