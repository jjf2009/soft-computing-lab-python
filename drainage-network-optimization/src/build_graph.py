"""Phase 3 (part 1) — Build the candidate-conduit graph from the SWMM model.

Everything the optimizer needs is grounded in the real .inp model so that f3 is a
genuine SWMM simulation:

  * nodes table            — id, x, y, invert elevation, type (junction/outfall/storage)
  * candidate_conduits     — id, from_node, to_node, length, slope, downhill_flag
                             (ROW ORDER = GENE ORDER of the chromosome)

For scope='sub' (fast demo) the graph is carved down to a connected sub-network of a
few hundred conduits upstream of one outfall. config.yaml switches sub <-> full.

    python -m src.build_graph            # writes outputs/tables/{nodes,candidate_conduits}.parquet
    python -m src.build_graph --full     # force the full network regardless of config
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque

import pandas as pd
import swmmio

from .common import find_inp, load_config, path


# --- defensive column lookup (swmmio column names vary across versions) ----------

def _col(df: pd.DataFrame, *keywords: str) -> str:
    """Return the first column whose name contains all given keywords (case-insensitive)."""
    for c in df.columns:
        cl = str(c).lower()
        if all(k.lower() in cl for k in keywords):
            return c
    raise KeyError(f"No column matching {keywords} in {list(df.columns)}")


def load_model() -> swmmio.Model:
    return swmmio.Model(str(find_inp()))


def nodes_table(model: swmmio.Model) -> pd.DataFrame:
    """Combine coordinates with node inverts and types into one table."""
    coords = model.inp.coordinates.copy()
    xcol, ycol = _col(coords, "x"), _col(coords, "y")
    coords = coords.rename(columns={xcol: "x", ycol: "y"})

    inverts: dict[str, float] = {}
    types: dict[str, str] = {}
    for section, label in [("junctions", "junction"), ("outfalls", "outfall"), ("storage", "storage")]:
        try:
            df = getattr(model.inp, section)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        try:
            inv = _col(df, "invert")
        except KeyError:
            inv = None
        for nid, row in df.iterrows():
            types[str(nid)] = label
            inverts[str(nid)] = float(row[inv]) if inv else math.nan

    rows = []
    for nid, row in coords.iterrows():
        nid = str(nid)
        rows.append(
            {
                "node_id": nid,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "elevation": inverts.get(nid, math.nan),
                "node_type": types.get(nid, "junction"),
            }
        )
    return pd.DataFrame(rows).set_index("node_id")


def conduits_table(model: swmmio.Model, nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per as-built conduit. Row order here fixes the gene order."""
    cond = model.inp.conduits.copy()
    inlet = _col(cond, "inlet") if _has(cond, "inlet") else _col(cond, "from")
    outlet = _col(cond, "outlet") if _has(cond, "outlet") else _col(cond, "to")
    try:
        length_col = _col(cond, "length")
    except KeyError:
        length_col = None

    rows = []
    for cid, row in cond.iterrows():
        a, b = str(row[inlet]), str(row[outlet])
        if a not in nodes.index or b not in nodes.index:
            # Endpoint missing coordinates: fall back to length-only, no geometry slope.
            length = float(row[length_col]) if length_col else math.nan
            slope = math.nan
            downhill = True
        else:
            na, nb = nodes.loc[a], nodes.loc[b]
            euclid = math.hypot(na["x"] - nb["x"], na["y"] - nb["y"])
            length = float(row[length_col]) if length_col and not math.isnan(row[length_col]) else euclid
            drop = (na["elevation"] - nb["elevation"])
            slope = drop / length if length else 0.0
            downhill = bool(drop >= 0) if not math.isnan(drop) else True
        rows.append(
            {
                "conduit_id": str(cid),
                "from_node": a,
                "to_node": b,
                "length": length,
                "slope": slope,
                "downhill_flag": downhill,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _has(df: pd.DataFrame, *keywords: str) -> bool:
    try:
        _col(df, *keywords)
        return True
    except KeyError:
        return False


def inlet_nodes(model: swmmio.Model) -> list[str]:
    """Nodes that receive subcatchment runoff (= the pipe-network inlets).

    These are the [SUBCATCHMENTS] outlets: the points where rain water enters the
    conduit network and therefore the nodes that MUST reach an outfall to drain.
    """
    try:
        subs = model.inp.subcatchments
    except Exception:
        return []
    if subs is None or len(subs) == 0:
        return []
    try:
        outlet_col = _col(subs, "outlet")
    except KeyError:
        return []
    return sorted({str(v) for v in subs[outlet_col].tolist()})


# --- sub-catchment carving -------------------------------------------------------

def select_subcatchment(nodes: pd.DataFrame, conduits: pd.DataFrame, cfg: dict):
    """Carve a connected slice upstream of one outfall, capped at max_conduits.

    Returns (nodes_sub, conduits_sub). BFS walks upstream (against flow direction)
    from the chosen outfall across the undirected graph until the conduit cap is hit.
    """
    sub = cfg["subcatchment"]
    cap = int(sub["max_conduits"])

    # Pick the outfall: configured id, else the one with the most upstream conduits.
    outfalls = nodes.index[nodes["node_type"] == "outfall"].tolist()
    if not outfalls:
        # No tagged outfall — use the node with highest in-degree as a proxy sink.
        indeg = conduits["to_node"].value_counts()
        outfalls = [indeg.index[0]] if len(indeg) else [nodes.index[0]]

    start = sub.get("outfall_id") or _largest_outfall(conduits, outfalls)

    # Undirected adjacency for the BFS.
    adj: dict[str, list[int]] = defaultdict(list)
    for i, row in conduits.iterrows():
        adj[row["from_node"]].append(i)
        adj[row["to_node"]].append(i)

    kept_conduits: list[int] = []
    seen_nodes = {start}
    q = deque([start])
    while q and len(kept_conduits) < cap:
        node = q.popleft()
        for ci in adj[node]:
            if ci in kept_conduits:
                continue
            kept_conduits.append(ci)
            if len(kept_conduits) >= cap:
                break
            row = conduits.loc[ci]
            for nb in (row["from_node"], row["to_node"]):
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    q.append(nb)

    conduits_sub = conduits.loc[kept_conduits].reset_index(drop=True)
    used_nodes = set(conduits_sub["from_node"]) | set(conduits_sub["to_node"])
    nodes_sub = nodes.loc[sorted(used_nodes & set(nodes.index))]
    return nodes_sub, conduits_sub


def _largest_outfall(conduits: pd.DataFrame, outfalls: list[str]) -> str:
    """Return the outfall reaching the most conduits by a quick undirected BFS."""
    adj: dict[str, set[str]] = defaultdict(set)
    for _, row in conduits.iterrows():
        adj[row["from_node"]].add(row["to_node"])
        adj[row["to_node"]].add(row["from_node"])
    best, best_n = outfalls[0], -1
    for o in outfalls:
        seen = {o}
        q = deque([o])
        while q:
            for nb in adj[q.popleft()]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if len(seen) > best_n:
            best, best_n = o, len(seen)
    return best


# --- entrypoint ------------------------------------------------------------------

def build(force_full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    model = load_model()
    nodes = nodes_table(model)
    conduits = conduits_table(model, nodes)
    all_inlets = set(inlet_nodes(model))

    scope = "full" if force_full else cfg.get("scope", "sub")
    if scope == "sub":
        nodes, conduits = select_subcatchment(nodes, conduits, cfg)

    in_scope = set(nodes.index)
    inlets = sorted(all_inlets & in_scope)
    outfalls = sorted(nodes.index[nodes["node_type"] == "outfall"].tolist())
    # Fallback: if no inlets map into scope, treat every in-scope node as needing drainage.
    if not inlets:
        inlets = sorted(in_scope - set(outfalls))

    meta = {
        "scope": scope,
        "n_nodes": int(len(nodes)),
        "n_conduits": int(len(conduits)),
        "inlets": inlets,
        "outfalls": outfalls,
    }

    tables = path("tables_dir")
    nodes.to_parquet(tables / "nodes.parquet")
    conduits.to_parquet(tables / "candidate_conduits.parquet")
    with open(tables / "graph_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if scope == "sub":
        # Carve a standalone, fast-to-simulate sub-model for this slice.
        from . import swmm_eval
        swmm_eval.write_subset_inp(
            set(nodes.index.astype(str)),
            set(conduits["conduit_id"].astype(str)),
            tables / "model_sub.inp",
        )
        print(f"  carved sub-model -> {tables / 'model_sub.inp'}")
    print(
        f"scope={scope}: {len(nodes)} nodes, {len(conduits)} candidate conduits "
        f"(genes), {len(inlets)} inlets, {len(outfalls)} outfalls. Tables -> {tables}"
    )
    return nodes, conduits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="force the full network")
    args = ap.parse_args()
    build(force_full=args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
