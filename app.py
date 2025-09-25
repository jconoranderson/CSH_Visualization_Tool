# app.py
# Flask app: upload CSV (raw "Details,Name" or already-parsed columns),
# parse & generate per-person PDFs (6-month blocks, newest first), with
# individual download links and "Download all as ZIP".
#
# HTML lives in templates/, CSS in static/. This file only serves routes and logic.

import io
import os
import re
import zipfile
from uuid import uuid4

# Headless backend for server environments
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from flask import Flask, render_template, request, send_file, abort, url_for

from datetime import datetime
from zoneinfo import ZoneInfo

app = Flask(__name__)

# --------------------------- Parsing utilities ---------------------------

TIME_NOW = datetime.now(ZoneInfo("America/New_York"))
DATE_TIME = TIME_NOW.strftime("%A, %B %d, %Y %I:%M:%S %p %Z%z")

DEFAULT_YEAR = datetime.now(ZoneInfo("America/New_York")).year

def _normalize_year(y):
    """2-digit years → 2000-based; leaves 4-digit years as-is."""
    try:
        y = int(y)
    except Exception:
        return None
    return y + 2000 if y < 100 else y

def parse_date_field(raw, default_year=DEFAULT_YEAR):
    """
    Normalize a 'Date:' field into 'M/D/YYYY'.
      - '5/21-5/22/2025'  -> '5/21/2025' (first day; year from right-hand date)
      - '5/20/25-5/21/25' -> '5/20/2025' (first date; 2-digit years normalized)
      - '5/21'            -> '5/21/<default_year>'
      - '5/21/25'         -> '5/21/2025'
    """
    if raw is None:
        return None
    s = str(raw).strip()
    s = re.sub(r"[–—−]", "-", s)  # normalize dash variants

    parts = [p.strip() for p in s.split("-") if p.strip()]
    first  = parts[0]
    second = parts[1] if len(parts) > 1 else None

    m1 = re.match(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})(?:/(?P<y>\d{2,4}))?$", first)
    if not m1:
        return None

    m = int(m1.group("m")); d = int(m1.group("d")); y = m1.group("y")

    if y is not None:
        y = _normalize_year(y)
    else:
        if second:
            m2 = re.search(r"/(?P<y>\d{2,4})\s*$", second)
            if m2:
                y = _normalize_year(m2.group("y"))
        if y is None:
            y = int(default_year)

    return f"{m}/{d}/{y}"

def _extract_time_and_period(line):
    """
    Extract ('HH:MM', 'AM'|'PM'|None) from a 'Start time...' or 'End time...' line.
    Prefers explicit checkbox markers like '( x ) PM'; else first AM/PM after the time.
    """
    if not line:
        return None, None

    t = re.search(r'(\d{1,2}:\d{2})', line)
    time_ = t.group(1) if t else None

    m_checked = re.search(r'[\(\[\{]\s*[xX]\s*[\)\]\}]\s*(A\.?M\.?|P\.?M\.?)', line, flags=re.I)
    if m_checked:
        tok = m_checked.group(1).upper().replace('.', '')
        return time_, ('AM' if tok.startswith('A') else 'PM')

    ampm_iter = list(re.finditer(r'\b(A\.?M\.?|P\.?M\.?)\b', line, flags=re.I))
    if ampm_iter:
        if time_:
            pos_time = line.find(time_)
            after = [m for m in ampm_iter if m.start() >= pos_time]
            m = after[0] if after else ampm_iter[0]
        else:
            m = ampm_iter[0]
        tok = m.group(1).upper().replace('.', '')
        return time_, ('AM' if tok.startswith('A') else 'PM')

    return time_, None

def clock_to_minutes(time_str, ampm):
    """'HH:MM' + 'AM'/'PM' -> minutes-from-midnight (0..1439) or None."""
    if not time_str or not ampm:
        return None
    try:
        h, m = time_str.strip().split(":")
        h, m = int(h), int(m)
    except Exception:
        return None
    ap = str(ampm).strip().upper().replace(".", "")
    if ap.startswith("A"):
        if h == 12: h = 0
    elif ap.startswith("P"):
        if h != 12: h += 12
    else:
        return None
    return (h * 60 + m) % (24 * 60)

def circular_mean_from_minutes_list(values):
    """Circular mean of minutes-of-day; values may include None/NaN."""
    vals = [v for v in (values or []) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    ang = 2 * np.pi * (arr / (24 * 60))
    a = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    if a < 0:
        a += 2 * np.pi
    mean_mins = a * (24 * 60) / (2 * np.pi)
    return int(round(mean_mins)) % (24 * 60)

def minutes_to_timestamp(mins):
    """Minutes-of-day -> Timestamp on arbitrary date for easy formatting."""
    if mins is None:
        return pd.NaT
    h = (mins // 60) % 24
    m = mins % 60
    return pd.Timestamp(2000, 1, 1, hour=int(h), minute=int(m))

def parse_details(txt):
    """Parse one raw 'Details' block into structured fields (incl. interruption averages)."""
    t = str(txt or "").replace("\r", "")

    # Date
    mdate = re.search(r"Date[:\s]*([^\n\r]+)", t, flags=re.I)
    date_str = parse_date_field(mdate.group(1)) if mdate else None

    # Main start/end (first pair)
    mstart = re.search(r"Start time[^\n]*", t, flags=re.I)
    mend   = re.search(r"End time[^\n]*",   t, flags=re.I)
    start_time, start_ampm = _extract_time_and_period(mstart.group(0) if mstart else "")
    end_time,   end_ampm   = _extract_time_and_period(mend.group(0)   if mend   else "")

    # Interruptions count (misspelling variants)
    mi = (re.search(r"INTERUPPTIONS TOTAL #\s*[:]*\s*(\d+)", t, re.I) or
          re.search(r"INTERRUPTIONS(?: TOTAL)?\s*#?\s*[:]*\s*(\d+)", t, re.I))
    interruptions = int(mi.group(1)) if mi else None

    # Totals
    mh = re.search(r"Hours[:\s]*([0-9]+)",   t, re.I)
    mm = re.search(r"Minutes[:\s]*([0-9]+)", t, re.I)
    hours   = int(mh.group(1)) if mh else None
    minutes = int(mm.group(1)) if mm else None

    # Interruption region (after header; else after first End)
    m_int_hdr = re.search(r"INTER+UP?TIONS?\s+TOTAL\s*#.*", t, flags=re.I)
    intr_region = t[m_int_hdr.end():] if m_int_hdr else (t[mend.end():] if mend else "")

    intr_start_mins_list, intr_end_mins_list = [], []
    if intr_region:
        starts = re.findall(r"Start time[^\n]*", intr_region, flags=re.I)
        ends   = re.findall(r"End time[^\n]*",   intr_region, flags=re.I)
        for s_line, e_line in zip(starts, ends):
            st, sap = _extract_time_and_period(s_line)
            et, eap = _extract_time_and_period(e_line)
            sm = clock_to_minutes(st, sap)
            em = clock_to_minutes(et, eap)
            if sm is not None and em is not None:
                intr_start_mins_list.append(sm)
                intr_end_mins_list.append(em)

    intr_start_mean_min = circular_mean_from_minutes_list(intr_start_mins_list)
    intr_end_mean_min   = circular_mean_from_minutes_list(intr_end_mins_list)

    return pd.Series({
        "date": date_str,            # 'M/D/YYYY'
        "start_time": start_time,
        "start_ampm": start_ampm,
        "end_time": end_time,
        "end_ampm": end_ampm,
        "interruptions": interruptions,
        "hours": hours,
        "minutes": minutes,
        # Interruption lists + per-note circular means
        "intr_start_mins_list": intr_start_mins_list if intr_start_mins_list else pd.NA,
        "intr_end_mins_list":   intr_end_mins_list   if intr_end_mins_list   else pd.NA,
        "intr_start_mean_min":  intr_start_mean_min,
        "intr_end_mean_min":    intr_end_mean_min,
        "intr_start_mean_ts":   minutes_to_timestamp(intr_start_mean_min),
        "intr_end_mean_ts":     minutes_to_timestamp(intr_end_mean_min),
    })

def _build_dt(date_str, time_str, ampm):
    if pd.isna(date_str) or pd.isna(time_str) or pd.isna(ampm):
        return pd.NaT
    for fmt in ("%m/%d/%Y %I:%M %p", "%m/%d/%y %I:%M %p"):
        try:
            return pd.to_datetime(f"{date_str} {time_str} {ampm}", format=fmt)
        except Exception:
            pass
    return pd.NaT

# ------------------- Future-date fix & windowing -------------------
def fix_future_dates_per_person(g: pd.DataFrame) -> pd.DataFrame:
    """Shift future start_dt/end_dt back by full years until <= today; ensure end>=start."""
    today = pd.Timestamp.today().normalize()
    person_name = getattr(g, "name", None)

    def _safe_replace_year(ts, y):
        try:
            return ts.replace(year=y)
        except ValueError:
            if ts.month == 2 and ts.day == 29:
                return ts.replace(year=y, day=28)
            raise

    def _to_past(ts):
        if pd.isna(ts): return ts
        out = ts
        while out.normalize() > today:
            out = _safe_replace_year(out, out.year - 1)
        return out

    g = g.copy()
    if person_name is not None and "Name" not in g.columns:
        g.insert(0, "Name", person_name)
    g["start_dt"] = g["start_dt"].apply(_to_past)
    g["end_dt"]   = g["end_dt"].apply(_to_past)
    mask = g["end_dt"].notna() & g["start_dt"].notna() & (g["end_dt"] < g["start_dt"])
    g.loc[mask, "end_dt"] = g.loc[mask, "end_dt"] + pd.Timedelta(days=1)
    return g

def build_windows_most_recent_first(person_df: pd.DataFrame):
    """Contiguous 6-month blocks, newest → oldest. Earliest block may be partial."""
    dates = pd.to_datetime(person_df["start_dt"]).dropna()
    if dates.empty: return []
    person_min = dates.min().normalize()
    person_max = dates.max().normalize()

    windows = []
    axis_end = person_max
    while axis_end >= person_min:
        axis_start = (axis_end - pd.DateOffset(months=6) + pd.Timedelta(days=1)).normalize()
        if axis_start < person_min:
            axis_start = person_min
        windows.append((axis_start, axis_end))
        axis_end = (axis_start - pd.Timedelta(days=1)).normalize()
    return windows

# ------------------- CSV loaders (raw or already-parsed) -------------------
def _coerce_parsed(df: pd.DataFrame):
    """If CSV already parsed (Name,start_dt,end_dt,duration_hr,interruptions[+intr_*]), coerce types."""
    needed = {"Name", "start_dt", "end_dt", "duration_hr", "interruptions"}
    if not needed.issubset(df.columns):
        return None
    out = df.copy()
    out["Name"] = out["Name"].fillna("Individual").astype(str).str.strip()
    out["start_dt"] = pd.to_datetime(out["start_dt"], errors="coerce")
    out["end_dt"]   = pd.to_datetime(out["end_dt"],   errors="coerce")
    out["duration_hr"] = pd.to_numeric(out["duration_hr"], errors="coerce")
    out["interruptions"] = pd.to_numeric(out["interruptions"], errors="coerce")
    out = out.groupby("Name", group_keys=False).apply(
        fix_future_dates_per_person,
        include_groups=False,
    )
    out = out.dropna(subset=["start_dt", "duration_hr"])
    return out

def _parse_raw_details(df: pd.DataFrame):
    """Parse raw 'Details,Name' CSV into normalized sleep DataFrame."""
    if not {"Details", "Name"}.issubset(df.columns):
        return None
    base = df[["Name", "Details"]].copy()
    base["Name"] = base["Name"].fillna("Individual").astype(str).str.strip()

    parsed = base["Details"].apply(parse_details)
    sleep = pd.concat([base[["Name"]], parsed], axis=1)

    # Build start_dt/end_dt
    sleep["start_dt"] = sleep.apply(lambda r: _build_dt(r["date"], r["start_time"], r["start_ampm"]), axis=1)
    sleep["end_dt"]   = sleep.apply(lambda r: _build_dt(r["date"], r["end_time"],   r["end_ampm"]),   axis=1)
    # If crosses midnight, bump end by 1 day
    mask = sleep["end_dt"].notna() & sleep["start_dt"].notna() & (sleep["end_dt"] < sleep["start_dt"])
    sleep.loc[mask, "end_dt"] += pd.Timedelta(days=1)

    # Duration: prefer totals; else compute from times
    has_total = sleep["hours"].notna() | sleep["minutes"].notna()
    sleep["duration_min"] = np.where(
        has_total,
        (sleep["hours"].astype("Float64").fillna(0) * 60 + sleep["minutes"].astype("Float64").fillna(0)),
        (sleep["end_dt"] - sleep["start_dt"]).dt.total_seconds() / 60.0
    )
    sleep["duration_hr"] = pd.to_numeric(sleep["duration_min"], errors="coerce") / 60.0

    # numeric interruptions
    sleep["interruptions"] = pd.to_numeric(sleep["interruptions"], errors="coerce")

    # Keep core + interruption features if present
    keep = ["Name", "start_dt", "end_dt", "duration_hr", "interruptions",
            "intr_start_mins_list", "intr_end_mins_list",
            "intr_start_mean_min", "intr_end_mean_min"]
    for col in keep:
        if col not in sleep.columns:
            sleep[col] = pd.NA

    # Fix future dates per person & drop invalids
    sleep = sleep.groupby("Name", group_keys=False).apply(
        fix_future_dates_per_person,
        include_groups=False,
    )
    sleep = sleep.dropna(subset=["start_dt", "duration_hr"])
    return sleep[keep]

def load_sleep_from_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Normalize common export column names
    df = df.rename(columns={"Progress Note Note": "Details", "Resident Name": "Name"})
    # Try already-parsed shape first
    out = _coerce_parsed(df)
    if out is not None:
        return out
    # Else parse raw Details
    out = _parse_raw_details(df)
    if out is not None:
        return out
    raise ValueError("CSV must contain either columns "
                     "[Name,start_dt,end_dt,duration_hr,interruptions] "
                     "or raw [Name,Details].")

# ------------------- Plotting helpers -------------------
def circular_mean_minutes_from_datetimes(dt_like: pd.Series):
    """Circular mean of time-of-day from a datetime series -> minutes-of-day."""
    s = pd.to_datetime(dt_like).dropna()
    if s.empty: return None
    mins = s.dt.hour * 60 + s.dt.minute + s.dt.second / 60.0
    ang = 2 * np.pi * (mins / (24 * 60))
    a = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    if a < 0: a += 2 * np.pi
    return int(round(a * (24 * 60) / (2 * np.pi))) % (24 * 60)

def add_minutes_circular(mins_from_midnight, delta_minutes):
    if mins_from_midnight is None or delta_minutes is None:
        return None
    return int((mins_from_midnight + delta_minutes) % (24 * 60))

def fmt_clock(mins):
    if mins is None:
        return "NA"
    h = (mins // 60) % 24
    m = mins % 60
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}"

def plot_person_windows_to_pdf(person: pd.DataFrame, pdf: PdfPages, person_name: str):
    windows = build_windows_most_recent_first(person)
    for (w_start, w_end) in windows:
        sub = person[(person["start_dt"] >= w_start) & (person["start_dt"] <= w_end)].sort_values("start_dt")
        if sub.empty: 
            continue

        y_all = pd.to_numeric(sub["duration_hr"], errors="coerce")
        mask  = y_all.notna() & sub["start_dt"].notna()
        x     = pd.to_datetime(sub.loc[mask, "start_dt"])
        y     = y_all.loc[mask]
        if x.empty:
            continue

        # Stats
        avg_hr   = y.mean()
        avg_intr = pd.to_numeric(sub.loc[mask, "interruptions"], errors="coerce").mean()
        start_min = circular_mean_minutes_from_datetimes(x)
        avg_min   = int(round(avg_hr * 60)) if pd.notna(avg_hr) else None
        end_min   = add_minutes_circular(start_min, avg_min)

        # Interruption averages (use list columns if present; else per-row means)
        intr_s_vals, intr_e_vals = [], []
        if "intr_start_mins_list" in sub.columns:
            for lst in sub.loc[mask, "intr_start_mins_list"].dropna():
                if isinstance(lst, (list, tuple, np.ndarray)):
                    intr_s_vals.extend([int(v) for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))])
        if "intr_end_mins_list" in sub.columns:
            for lst in sub.loc[mask, "intr_end_mins_list"].dropna():
                if isinstance(lst, (list, tuple, np.ndarray)):
                    intr_e_vals.extend([int(v) for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))])
        if not intr_s_vals and "intr_start_mean_min" in sub.columns:
            intr_s_vals = pd.to_numeric(sub.loc[mask, "intr_start_mean_min"], errors="coerce").dropna().astype(int).tolist()
        if not intr_e_vals and "intr_end_mean_min" in sub.columns:
            intr_e_vals = pd.to_numeric(sub.loc[mask, "intr_end_mean_min"], errors="coerce").dropna().astype(int).tolist()

        intr_start_mean = circular_mean_from_minutes_list(intr_s_vals) if intr_s_vals else None
        intr_end_mean   = circular_mean_from_minutes_list(intr_e_vals) if intr_e_vals else None

        # Strings
        if pd.notna(avg_hr):
            H = int(avg_hr); M = int(round((avg_hr - H) * 60)) % 60
            avg_str = f"{avg_hr:.2f} h ({H}h {M:02d}m)"
        else:
            avg_str = "NA"
        s_str = fmt_clock(start_min)
        e_str = fmt_clock(end_min)
        i_str = f"{avg_intr:.2f}" if pd.notna(avg_intr) else "NA"
        intr_s_str = fmt_clock(intr_start_mean)
        intr_e_str = fmt_clock(intr_end_mean)

        # Figure
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=False)
        ax.plot(x, y, marker="o")

        # Trend line
        y_np = y.to_numpy()
        finite = np.isfinite(y_np)
        if finite.sum() >= 2 and np.unique(y_np[finite]).size > 1:
            xnum = mdates.date2num(x)
            slope, intercept = np.polyfit(xnum, y_np, 1)
            xfit = np.array([xnum.min(), xnum.max()])
            yfit = slope * xfit + intercept
            ax.plot(mdates.num2date(xfit), yfit, linestyle="--", linewidth=1.5)

        # Average line
        if pd.notna(avg_hr):
            ax.axhline(avg_hr, linestyle=":", linewidth=1.2, color="black", alpha=0.6)

        # X-axis = observed span
        obs_min = x.min().normalize()
        obs_max = x.max().normalize()
        if obs_min == obs_max:
            obs_min -= pd.Timedelta(days=3)
            obs_max += pd.Timedelta(days=3)
        ax.set_xlim(obs_min, obs_max)
        span_days = (obs_max - obs_min).days
        if span_days <= 60:
            step = max(1, span_days // 8)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        else:
            months = max(1, int(round(span_days / 30)))
            interval = 1 if months <= 6 else 2 if months <= 12 else 3
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        period_label = f"{obs_min.strftime('%b %d, %Y')} – {obs_max.strftime('%b %d, %Y')}"

        # Label
        ax.text(0.98, 1.35,
                f"Avg sleep: {avg_str}\n"
                f"Avg start: {s_str}\n"
                f"Avg end:   {e_str}\n"
                f"Avg interruptions: {i_str}\n"
                f"Avg intr. start: {intr_s_str}\n"
                f"Avg intr. end:   {intr_e_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)

        ax.set_title(f"Sleep duration — {person_name}  |  {period_label}")
        ax.set_xlabel("Date"); ax.set_ylabel("Duration (hours)")
        fig.autofmt_xdate()

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# ------------------- Storage for generated PDFs -------------------
# Use the directory of this file as an anchor so paths are stable
# when running under WSGI/daemonized servers.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_ROOT = os.path.join(BASE_DIR, "generated_pdfs")
os.makedirs(STORE_ROOT, exist_ok=True)

# ------------------- Flask routes -------------------
@app.route("/", methods=["GET"])
def index():
    # If you added theme support in base.html, pass a theme dict here.
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "csv" not in request.files:
        abort(400, "No file part named 'csv'.")
    f = request.files["csv"]
    if not f or f.filename == "":
        abort(400, "No file selected.")
    if not f.filename.lower().endswith(".csv"):
        abort(400, "Please upload a .csv file.")

    raw = f.read()
    try:
        sleep = load_sleep_from_csv(raw)   # handles raw or already-parsed formats
    except Exception as e:
        abort(400, f"Failed to parse CSV: {e}")

    # alphabetical order
    names = sorted(sleep["Name"].dropna().astype(str).unique(), key=lambda s: s.casefold())

    # unique job folder for this upload
    job_id = uuid4().hex
    job_dir = os.path.join(STORE_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # create per-person PDFs on disk
    files = []
    for name in names:
        person = sleep[sleep["Name"] == name].copy()
        if person["start_dt"].dropna().empty:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "Individual"
        pdf_path = os.path.join(job_dir, f"{safe}.pdf")
        with PdfPages(pdf_path) as pdf:
            info = pdf.infodict()
            info["Title"] = f"Sleep Charts — {name}"
            info["Author"] = "Sleep Plotter"
            info["Subject"] = "Per-person charts (6-month blocks, newest first)"
            plot_person_windows_to_pdf(person, pdf, name)
        files.append(f"{safe}.pdf")

    if not files:
        abort(400, "No valid records to plot.")

    return render_template("results.html", job_id=job_id, files=files, n=len(files))

@app.route("/download/<job_id>/<path:filename>", methods=["GET"])
def download_pdf(job_id, filename):
    # simple path safety
    if "/" in filename or ".." in filename:
        abort(400, "Invalid filename.")
    job_dir = os.path.join(STORE_ROOT, job_id)
    full_path = os.path.join(job_dir, filename)
    if not os.path.isfile(full_path):
        abort(404, "File not found.")
    return send_file(full_path, mimetype="application/pdf", as_attachment=True, download_name=filename)

@app.route("/download-all/<job_id>", methods=["GET"])
def download_all_zip(job_id):
    job_dir = os.path.join(STORE_ROOT, job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Unknown job.")

    pdf_files = [fn for fn in os.listdir(job_dir) if fn.lower().endswith(".pdf")]
    pdf_files.sort(key=str.casefold)
    if not pdf_files:
        abort(404, "No PDFs to bundle.")

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in pdf_files:
            full = os.path.join(job_dir, fn)
            zf.write(full, arcname=fn)
    mem_zip.seek(0)

    return send_file(
        mem_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"CSH_Sleep_Charts{DATE_TIME}.zip",
    )

@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    job_dir = os.path.join(STORE_ROOT, job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Unknown job.")
    files = [fn for fn in os.listdir(job_dir) if fn.lower().endswith(".pdf")]
    files.sort(key=str.casefold)
    if not files:
        abort(404, "No files for this job.")
    return render_template("results.html", job_id=job_id, files=files, n=len(files))

if __name__ == "__main__":
    # Run on all interfaces for intranet access
    app.run(host="0.0.0.0", port=5001, debug=False)
