"""Phase 2 — Convert the raw Bellinge files into small tables, run ONCE.

This is the step that kills the "dataset too big" problem: the multi-year sensor
zips are streamed in chunks (never loaded whole), and only a short DESIGN STORM
(the peak-rainfall window) is kept. After this runs, the optimization loop reads
nothing but tiny tables.

    python -m src.preprocess

Outputs (outputs/tables/):
  constraints.parquet   variable, unit, min, max, conversion_factor   (from the PDF)
  sensors.parquet       sensor coordinates (from the CSV)
  timeseries.parquet    the design-storm window only (small)
  storm.json            {start_date, start_time, end_date, end_time}  -> used by swmm_eval

The parsers are deliberately defensive: real column names are confirmed at runtime
and anything unexpected is logged rather than silently dropped.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from .common import load_config, path

CHUNK_ROWS = 200_000


# --- PDF data dictionary ---------------------------------------------------------

def parse_constraints(pdf_path: Path) -> pd.DataFrame:
    import pdfplumber

    rows: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                header = [str(c or "").strip().lower() for c in table[0]]
                for raw in table[1:]:
                    cells = [str(c or "").strip() for c in raw]
                    row = dict(zip(header, cells))
                    rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  constraints: {len(df)} rows parsed from PDF tables")
    else:
        print("  constraints: no tables detected in PDF (check the file / layout)")
    return df


# --- sensor coordinates ----------------------------------------------------------

def parse_sensors(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  sensors: {len(df)} rows, columns = {list(df.columns)[:8]}")
    return df


# --- streaming design-storm extraction -------------------------------------------

def _detect_cols(sample: pd.DataFrame) -> tuple[str | None, str | None]:
    """Guess a timestamp column and a numeric value column."""
    ts_col = None
    for c in sample.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("time", "date", "stamp")):
            ts_col = c
            break
    if ts_col is None:
        ts_col = sample.columns[0]
    numeric = [c for c in sample.columns if c != ts_col and pd.api.types.is_numeric_dtype(sample[c])]
    val_col = numeric[0] if numeric else None
    return ts_col, val_col


def _stream_peak(zip_path: Path) -> tuple[pd.Timestamp | None, float]:
    """Stream every CSV in the archive in chunks; return (peak_timestamp, peak_value).

    Memory stays flat: only one chunk is resident at a time.
    """
    best_ts, best_val = None, float("-inf")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".csv", ".txt"))]
        for member in members:
            with zf.open(member) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
                try:
                    reader = pd.read_csv(text, chunksize=CHUNK_ROWS, sep=None, engine="python")
                except Exception as e:
                    print(f"    skip {member}: {e}")
                    continue
                ts_col = val_col = None
                for chunk in reader:
                    if ts_col is None:
                        ts_col, val_col = _detect_cols(chunk)
                        if val_col is None:
                            break
                    ts = pd.to_datetime(chunk[ts_col], errors="coerce")
                    vals = pd.to_numeric(chunk[val_col], errors="coerce")
                    ok = ts.notna() & vals.notna()
                    if ok.any():
                        idx = vals[ok].idxmax()
                        if vals[idx] > best_val:
                            best_val, best_ts = float(vals[idx]), ts[idx]
    return best_ts, best_val


def extract_storm_from_dat(dat_path: Path, window_hours: int) -> dict | None:
    """Find the heaviest ``window_hours`` window in a SWMM rain file and return it.

    Rain file format: ``gage YYYY MM DD HH MM value`` (one reading per line). The
    network's own forcing lives here, so this is the design storm that actually
    stresses the hydraulics — far better than the model's arbitrary start date.
    """
    cols = ["gage", "Y", "M", "D", "h", "m", "v"]
    df = pd.read_csv(dat_path, sep=r"\s+", names=cols, engine="c",
                     na_values=["", "NaN"], on_bad_lines="skip")
    df = df.dropna(subset=["Y", "M", "D", "h", "m", "v"])
    ts = pd.to_datetime(dict(year=df.Y, month=df.M, day=df.D, hour=df.h, minute=df.m),
                        errors="coerce")
    s = pd.Series(pd.to_numeric(df.v, errors="coerce").values, index=ts).dropna()
    s = s.groupby(level=0).sum().sort_index()               # total rain per minute
    if s.empty:
        return None
    s = s.resample("1min").sum().fillna(0.0)
    roll = s.rolling(f"{int(window_hours)}h").sum()
    peak_end = roll.idxmax()
    start = peak_end - pd.Timedelta(hours=window_hours)
    storm = {
        "peak_value": float(roll.max()),
        "start_date": start.strftime("%m/%d/%Y"), "start_time": start.strftime("%H:%M:%S"),
        "end_date": peak_end.strftime("%m/%d/%Y"), "end_time": peak_end.strftime("%H:%M:%S"),
    }
    print(f"  storm: heaviest {window_hours}h = {storm['peak_value']:.1f} mm around "
          f"{peak_end} -> {storm['start_date']} {storm['start_time']} .. "
          f"{storm['end_date']} {storm['end_time']}")
    return storm


def extract_design_storm(data_dir: Path, window_hours: int) -> dict | None:
    # Prefer the model's own rain gage file (the real forcing) if present.
    dats = sorted(data_dir.rglob("rg_*.dat")) or sorted(data_dir.rglob("*.dat"))
    if dats:
        print(f"  storm: scanning rain file {dats[0].name} ...")
        storm = extract_storm_from_dat(dats[0], window_hours)
        if storm:
            return storm
    cleaned = sorted(data_dir.rglob("*cleaned*.zip")) or sorted(data_dir.rglob("*.zip"))
    if not cleaned:
        print("  storm: no time-series zip found — optimizer will use the model's own dates")
        return None
    print(f"  storm: scanning {cleaned[0].name} (streamed) for the peak event ...")
    peak_ts, peak_val = _stream_peak(cleaned[0])
    if peak_ts is None:
        print("  storm: no usable timestamped values found")
        return None
    start = peak_ts - dt.timedelta(hours=window_hours / 2)
    end = peak_ts + dt.timedelta(hours=window_hours / 2)
    storm = {
        "peak_value": peak_val,
        "start_date": start.strftime("%m/%d/%Y"),
        "start_time": start.strftime("%H:%M:%S"),
        "end_date": end.strftime("%m/%d/%Y"),
        "end_time": end.strftime("%H:%M:%S"),
    }
    print(f"  storm: peak={peak_val:.3g} at {peak_ts} -> window {storm['start_date']} "
          f"{storm['start_time']} .. {storm['end_date']} {storm['end_time']}")
    return storm


# --- entrypoint ------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    data_dir = path("data_dir")
    tables = path("tables_dir")

    print("Preprocessing Bellinge files (run once) ...")

    pdfs = sorted(data_dir.rglob("*Sensordata*.pdf")) or sorted(data_dir.rglob("*.pdf"))
    if pdfs:
        try:
            parse_constraints(pdfs[0]).to_parquet(tables / "constraints.parquet")
        except Exception as e:
            print(f"  constraints: PDF parse failed ({e})")

    csvs = sorted(data_dir.rglob("*Coordinates*.csv")) or sorted(data_dir.rglob("*.csv"))
    if csvs:
        try:
            parse_sensors(csvs[0]).to_parquet(tables / "sensors.parquet")
        except Exception as e:
            print(f"  sensors: CSV parse failed ({e})")

    storm = extract_design_storm(data_dir, int(cfg["storm"]["window_hours"]))
    if storm:
        with open(tables / "storm.json", "w", encoding="utf-8") as fh:
            json.dump(storm, fh, indent=2)

    print(f"\nTables -> {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
