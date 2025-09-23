Sleep Plotter — Deployment
==========================

Quick start (local dev)
-----------------------
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r CSH_Sleep_Visualization_Tool/requirements.txt`
- Run dev server: `python CSH_Sleep_Visualization_Tool/app.py`
- Open http://localhost:5000

Production (Gunicorn)
---------------------
This project includes a WSGI entrypoint at `CSH_Sleep_Visualization_Tool/wsgi.py` and a package marker `__init__.py`.

From the repo root:

1) Create a virtualenv and install dependencies
   `python3 -m venv .venv && source .venv/bin/activate`
   `pip install -r CSH_Sleep_Visualization_Tool/requirements.txt`

2) Run Gunicorn
   `gunicorn 'CSH_Sleep_Visualization_Tool.wsgi:app' --workers 2 --bind 0.0.0.0:8000`

3) Optional: behind Nginx (recommended)
   - Point Nginx to proxy `http://127.0.0.1:8000`.
   - Serve static files (CSS) via Flask or directly via Nginx from `CSH_Sleep_Visualization_Tool/static`.

Notes
-----
- Generated PDFs are saved under `CSH_Sleep_Visualization_Tool/generated_pdfs` (path anchored to the app directory).
- The app accepts two CSV formats: raw `Name,Details` or parsed `Name,start_dt,end_dt,duration_hr,interruptions`.
- For small deployments, Flask can serve static files. For larger deployments, offload `/static` to Nginx.
