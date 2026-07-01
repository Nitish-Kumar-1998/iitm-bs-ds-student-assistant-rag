"""
main.py
-------
Thin entry-point wrapper, located inside app/.

This file contains NO logic of its own — it exists purely so that
existing commands, deployment scripts, and muscle memory from v1 keep
working with minimal changes:

    uvicorn app.main:app --reload --port 8000
    uvicorn app.main:app --host 0.0.0.0 --port $PORT   (Render start command)

The real FastAPI app lives in app/api/app.py — see that file (and
app/api/README.md) for the actual implementation, endpoint contracts,
and pipeline wiring.
"""

from app.api.app import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    from app.api.config import HOST, PORT, validate

    validate()
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)