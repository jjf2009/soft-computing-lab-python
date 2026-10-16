"""Phase 4 (part 1) — Decode a chromosome into a SWMM run and read its flooding.

Cost-control strategy (f3 is the expensive objective):
  * The baseline .inp is parsed ONCE into sections. Per evaluation we only filter out
    the dropped conduits (a cheap text transform) and shorten the simulation to the
    design-storm window — no per-eval swmmio rewrite.
  * Results are cached by the active-conduit bitset: identical layouts are never
    re-simulated (in-memory per process + an optional shared sqlite disk cache).

Returned metrics (all >= 0, to be minimised): total flood volume, number of flooded
nodes, and total hours flooded (a surcharge-duration proxy).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass

import numpy as np

from .common import find_inp, load_config, path
from . import encoding


# --- lightweight .inp section manager -------------------------------------------

# Sections keyed by a LINK name. XSECTIONS/LOSSES also hold regulator (weir/orifice/
# pump/outlet) rows, which must be kept even when a conduit is dropped.
_LINK_KEYED_SECTIONS = {"XSECTIONS", "LOSSES"}


@dataclass
class BaselineINP:
    header: str                       # text before the first section
    sections: list[tuple[str, list[str]]]  # (NAME, raw lines) in order
    storm: dict | None                # {start_date,start_time,end_date,end_time} or None
    conduit_ids: set[str]             # every [CONDUITS] id (to tell conduits from regulators)
    ext_files: dict[str, str]         # sidecar data file (e.g. rain .dat) -> absolute path

    @staticmethod
    def _first_token(line: str) -> str | None:
        s = line.strip()
        if not s or s.startswith(";"):
            return None
        return s.split()[0]

    def write_variant(self, active_ids: set[str], out_path: str) -> None:
        """Write the baseline model with inactive CONDUITS removed. All nodes and
        regulators stay, so the model is always valid (no ERROR 209 / 143)."""
        cfg = load_config()
        step = int(cfg["storm"]["routing_step_seconds"])
        chunks: list[str] = [self.header]
        for name, lines in self.sections:
            up = name.upper()
            chunks.append(f"[{name}]\n")
            if up == "CONDUITS":
                for ln in lines:
                    tok = self._first_token(ln)
                    if tok is None or tok in active_ids:
                        chunks.append(ln)
            elif up in _LINK_KEYED_SECTIONS:
                for ln in lines:
                    tok = self._first_token(ln)
                    # drop only rows for a conduit that is currently inactive;
                    # keep regulator rows (tok not a conduit) and active conduits.
                    if tok is None or tok not in self.conduit_ids or tok in active_ids:
                        chunks.append(ln)
            elif up == "OPTIONS":
                chunks.extend(self._options_with_window(lines, step))
            else:
                chunks.extend(lines)
        text = "".join(chunks)
        # Resolve sidecar data files (e.g. the rain .dat) to absolute paths so the
        # variant runs from any temp directory.
        for bare, absolute in self.ext_files.items():
            text = text.replace(f'"{bare}"', f'"{absolute}"')
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _options_with_window(self, lines: list[str], step: int) -> list[str]:
        """Rewrite OPTIONS so every simulation covers only a short window (the design
        storm if preprocess found one, otherwise window_hours from the model's own
        start) and uses the configured routing step. This caps the cost of each run."""
        import datetime as dt

        cfg = load_config()
        window_hours = float(cfg["storm"]["window_hours"])

        opts = {self._first_token(ln).upper(): ln.split(None, 1)[1].strip()
                for ln in lines if self._first_token(ln)}

        if self.storm:
            start_date, start_time = self.storm["start_date"], self.storm["start_time"]
            end_date, end_time = self.storm["end_date"], self.storm["end_time"]
        else:
            start_date = opts.get("START_DATE", "01/01/2020")
            start_time = opts.get("START_TIME", "00:00:00")
            try:
                t0 = dt.datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")
            except ValueError:
                t0 = dt.datetime.strptime(start_date, "%m/%d/%Y")
            t1 = t0 + dt.timedelta(hours=window_hours)
            end_date, end_time = t1.strftime("%m/%d/%Y"), t1.strftime("%H:%M:%S")

        overrides = {
            "START_DATE": start_date, "START_TIME": start_time,
            "REPORT_START_DATE": start_date, "REPORT_START_TIME": start_time,
            "END_DATE": end_date, "END_TIME": end_time,
            "ROUTING_STEP": str(step),
        }
        seen, out = set(), []
        for ln in lines:
            tok = self._first_token(ln)
            key = tok.upper() if tok else None
            if key in overrides:
                out.append(f"{tok:<21}{overrides[key]}\n")
                seen.add(key)
            else:
                out.append(ln)
        for key, val in overrides.items():
            if key not in seen:
                out.append(f"{key:<21}{val}\n")
        return out


def _parse_inp(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    header_lines: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for raw in text.splitlines(keepends=True):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip()
            current = []
            sections.append((name, current))
        elif current is None:
            header_lines.append(raw)
        else:
            current.append(raw)
    return "".join(header_lines), sections


def _load_storm() -> dict | None:
    """Design-storm window written by preprocess.py, if available."""
    p = path("tables_dir") / "storm.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


_BASELINE: BaselineINP | None = None

# Section classification for carving a standalone sub-model.
_NODE_SECTIONS = {"JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS", "COORDINATES",
                  "DWF", "INFLOWS", "RDII", "TREATMENT"}           # keyed by node name
_LINK_KEYED = {"XSECTIONS", "LOSSES", "VERTICES"}                  # keyed by link name
_LINK_PAIR_SPECIAL = {"PUMPS", "ORIFICES", "WEIRS", "OUTLETS"}     # Name Node1 Node2 ...
_SUBCATCH_SECTIONS = {"SUBCATCHMENTS", "SUBAREAS", "INFILTRATION",
                      "COVERAGES", "LOADINGS", "GROUNDWATER"}      # keyed by subcatch name
_DROP_SECTIONS = {"CONTROLS", "TAGS"}                             # reference objects that may be gone


def model_inp_path():
    """The .inp the simulator runs: the carved sub-model for scope='sub' (if built),
    otherwise the full Bellinge model."""
    if load_config().get("scope") == "sub":
        sub = path("tables_dir") / "model_sub.inp"
        if sub.exists():
            return sub
    return find_inp()


def baseline() -> BaselineINP:
    global _BASELINE
    if _BASELINE is None:
        inp_path = model_inp_path()
        text = inp_path.read_text(encoding="utf-8", errors="ignore")
        header, sections = _parse_inp(text)
        conduit_ids = set()
        for name, lines in sections:
            if name.upper() == "CONDUITS":
                for ln in lines:
                    tok = BaselineINP._first_token(ln)
                    if tok is not None:
                        conduit_ids.add(tok)
        # Sidecar data files live next to the model (rain .dat, etc.). The full model's
        # directory holds them even when we simulate a carved sub-model.
        model_dir = find_inp().parent
        skip_ext = {".inp", ".rpt", ".out"}
        ext_files = {f.name: str(f.resolve()) for f in model_dir.iterdir()
                     if f.is_file() and f.suffix.lower() not in skip_ext}
        _BASELINE = BaselineINP(header=header, sections=sections, storm=_load_storm(),
                                conduit_ids=conduit_ids, ext_files=ext_files)
    return _BASELINE


def write_subset_inp(keep_nodes: set[str], keep_conduits: set[str], out_path) -> None:
    """Carve a standalone, runnable sub-model: keep only the given nodes and conduits
    (plus pumps/weirs/orifices/outlets fully inside the slice, the subcatchments
    draining into it, and all global sections such as rain and options).

    Used by build_graph for scope='sub' so the demo simulates a small, fast network
    rather than the full model with most pipes stripped out. Every section that names
    a node or link is filtered so no line references a removed object (SWMM ERROR 209).
    """
    def first_token(line: str):
        s = line.strip()
        return None if (not s or s.startswith(";")) else s.split()[0]

    text = find_inp().read_text(encoding="utf-8", errors="ignore")
    header, sections = _parse_inp(text)

    # Pass 1: special links fully inside the slice, and subcatchments draining into it.
    kept_links: set[str] = set(keep_conduits)
    kept_subcatch: set[str] = set()
    for name, lines in sections:
        up = name.upper()
        if up in _LINK_PAIR_SPECIAL:
            for ln in lines:
                tok = first_token(ln)
                parts = ln.split()
                if tok and len(parts) >= 3 and parts[1] in keep_nodes and parts[2] in keep_nodes:
                    kept_links.add(tok)
        elif up == "SUBCATCHMENTS":
            for ln in lines:
                tok = first_token(ln)
                parts = ln.split()
                if tok and len(parts) >= 3 and parts[2] in keep_nodes:  # Name Raingage Outlet
                    kept_subcatch.add(tok)

    # Pass 2: write filtered sections.
    chunks = [header]
    for name, lines in sections:
        up = name.upper()
        chunks.append(f"[{name}]\n")
        if up in _DROP_SECTIONS:
            continue  # keep the header, drop the body
        if up in _NODE_SECTIONS:
            keep = keep_nodes
        elif up == "CONDUITS":
            keep = keep_conduits
        elif up in _LINK_KEYED:
            keep = kept_links
        elif up in _LINK_PAIR_SPECIAL:
            keep = kept_links
        elif up in _SUBCATCH_SECTIONS:
            keep = kept_subcatch
        else:
            chunks.extend(lines)  # global section, keep verbatim
            continue
        for ln in lines:
            tok = first_token(ln)
            if tok is None or tok in keep:
                chunks.append(ln)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(chunks))


# --- flooding read ---------------------------------------------------------------

def _parse_rpt_flooding(rpt_path: str) -> tuple[float, int, float]:
    """Parse the 'Node Flooding Summary' table of a SWMM .rpt report.

    Returns (total_flood_volume, flooded_node_count, total_hours_flooded).
    Columns: Node  HoursFlooded  MaxRate  Day  hr:min  TotalFloodVolume  MaxPonded.
    """
    total_vol = 0.0
    hours_sum = 0.0
    count = 0
    try:
        with open(rpt_path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return 0.0, 0, 0.0

    in_section = False
    for ln in lines:
        s = ln.strip()
        if "Node Flooding Summary" in s:
            in_section = True
            continue
        if not in_section:
            continue
        if s.startswith("***") or "Outfall Loading Summary" in s or "Storage Volume Summary" in s:
            break
        if not s or s.startswith("-") or s.startswith("Node") or ":" not in s and not s[0:1].isalnum():
            # header/separator rows
            if "No nodes were flooded" in s:
                break
            continue
        parts = s.split()
        # must contain a time token "hr:min"
        time_idx = next((i for i, t in enumerate(parts) if ":" in t), None)
        if time_idx is None or time_idx < 3:
            continue
        nums = []
        for t in parts:
            if ":" in t:
                continue
            try:
                nums.append(float(t))
            except ValueError:
                pass
        # nums = [hours, maxrate, day, floodvol, ponded]
        if len(nums) >= 4:
            hours_sum += nums[0]
            total_vol += nums[-2]
            count += 1
    return total_vol, count, hours_sum


# --- public API ------------------------------------------------------------------

@dataclass
class FloodResult:
    flood_volume: float
    flooded_nodes: int
    hours_flooded: float
    failed: bool = False  # True if SWMM errored (treated as maximally bad)


def _bits_key(bits: np.ndarray) -> str:
    return hashlib.blake2b(np.asarray(bits, dtype=np.int8).tobytes(), digest_size=16).hexdigest()


class _DiskCache:
    """Best-effort shared cache across worker processes."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            con = sqlite3.connect(db_path, timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS sim (k TEXT PRIMARY KEY, vol REAL, nodes INT, hrs REAL, failed INT)"
            )
            con.commit()
            con.close()
        except Exception:
            pass

    def get(self, key: str) -> FloodResult | None:
        try:
            con = sqlite3.connect(self.db_path, timeout=30)
            row = con.execute("SELECT vol,nodes,hrs,failed FROM sim WHERE k=?", (key,)).fetchone()
            con.close()
            if row:
                return FloodResult(row[0], int(row[1]), row[2], bool(row[3]))
        except Exception:
            pass
        return None

    def put(self, key: str, r: FloodResult) -> None:
        try:
            con = sqlite3.connect(self.db_path, timeout=30)
            con.execute(
                "INSERT OR IGNORE INTO sim VALUES (?,?,?,?,?)",
                (key, r.flood_volume, r.flooded_nodes, r.hours_flooded, int(r.failed)),
            )
            con.commit()
            con.close()
        except Exception:
            pass


_MEM_CACHE: dict[str, FloodResult] = {}
_DISK: _DiskCache | None = None


def _disk() -> _DiskCache:
    global _DISK
    if _DISK is None:
        _DISK = _DiskCache(str(path("cache_dir") / "sim_cache.sqlite"))
    return _DISK


def simulate(bits: np.ndarray, use_cache: bool = True) -> FloodResult:
    """Run SWMM for the layout encoded by ``bits`` and return its flooding metrics."""
    key = _bits_key(bits)
    if use_cache:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]
        hit = _disk().get(key)
        if hit is not None:
            _MEM_CACHE[key] = hit
            return hit

    g = encoding.graph()
    active_ids = {g.conduits.iloc[i]["conduit_id"] for i in encoding.active_indices(bits)}

    result = _run_swmm(active_ids)
    if use_cache:
        _MEM_CACHE[key] = result
        _disk().put(key, result)
    return result


def _run_swmm(active_ids: set[str]) -> FloodResult:
    from pyswmm import Simulation

    with tempfile.TemporaryDirectory(dir=path("cache_dir")) as tmp:
        inp = os.path.join(tmp, "variant.inp")
        rpt = os.path.join(tmp, "variant.rpt")
        out = os.path.join(tmp, "variant.out")
        baseline().write_variant(active_ids, inp)
        try:
            # Iterate rather than sim.execute(): execute() aborts the whole process on
            # a SWMM input error, while iterating raises a catchable Python exception.
            with Simulation(inp, rpt, out) as sim:
                for _ in sim:
                    pass
            vol, nodes, hrs = _parse_rpt_flooding(rpt)
            return FloodResult(vol, nodes, hrs, failed=False)
        except Exception:
            # An un-runnable network is maximally bad on f3 (the feasibility gate and
            # repair should keep these rare).
            return FloodResult(1e12, 10**6, 1e9, failed=True)


def baseline_result() -> FloodResult:
    """f3 of the unmodified network (all conduits active) — the reference design."""
    g = encoding.graph()
    all_ids = set(g.conduits["conduit_id"].tolist())
    return _run_swmm(all_ids)
