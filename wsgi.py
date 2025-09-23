"""WSGI entrypoint for production servers (e.g., Gunicorn).

Usage (from repo root):
  gunicorn 'CSH_Sleep_Visualization_Tool.wsgi:app' --workers 2 --bind 0.0.0.0:8000
"""

from .app import app  # re-export Flask app instance for WSGI servers

