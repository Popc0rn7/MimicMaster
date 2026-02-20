"""Main entry point for Mimic Master application."""

import uvicorn

from mimic_master.api.app import app
from mimic_master.config import settings


def main() -> None:
    """Run the FastAPI application with Uvicorn server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Enable auto-reload during development
    )


if __name__ == "__main__":
    main()
