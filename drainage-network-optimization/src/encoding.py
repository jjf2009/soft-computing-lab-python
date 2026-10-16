"""Phase 3 (part 2) — Binary encoding, decoding and the repair operator.

Chromosome: one bit per row of candidate_conduits (row order = gene order).
  bit = 1 -> conduit kept,  bit = 0 -> conduit dropped.
Active nodes are derived, not stored: a node with no active conduit is "removed".

The repair operator keeps the population feasible: if any inlet (a node that
receives subcatchment runoff) is cut off from every outfall, it re-enables the
shortest downhill-preferring path that restores drainage. This avoids spending
expensive SWMM simulations on networks that cannot drain at all.
"""
from __future__ import annotations

import functools
import heapq
import json
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import load_config, path

UPHILL_WEIGHT_FACTOR = 50.0  # uphill edges cost more during repair -> gravity preferred


@dataclass
class GraphData:
    conduits: pd.DataFrame          # row order = gene order
    n_genes: int
    endpoints: list[tuple[str, str]]
    lengths: np.ndarray
    downhill: np.ndarray            # bool per gene
    node_adj: dict[str, list[int]]  # node -> gene indices touching it
    inlets: list[str]
    outfalls: list[str]


@functools.lru_cache(maxsize=1)
def graph() -> GraphData:
    """Load the candidate graph once (cached per process)."""
    tables = path("tables_dir")
    conduits = pd.read_parquet(tables / "candidate_conduits.parquet")
    with open(tables / "graph_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    endpoints = list(zip(conduits["from_node"].astype(str), conduits["to_node"].astype(str)))
    lengths = conduits["length"].to_numpy(dtype=float)
    lengths = np.nan_to_num(lengths, nan=float(np.nanmax(lengths) if len(lengths) else 1.0))
    downhill = conduits["downhill_flag"].to_numpy(dtype=bool)

    node_adj: dict[str, list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(endpoints):
        node_adj[a].append(i)
        node_adj[b].append(i)

    return GraphData(
        conduits=conduits,
        n_genes=len(conduits),
        endpoints=endpoints,
        lengths=lengths,
        downhill=downhill,
        node_adj=node_adj,
        inlets=[str(n) for n in meta["inlets"]],
        outfalls=[str(n) for n in meta["outfalls"]],
    )


def n_genes() -> int:
    return graph().n_genes


# --- decoding --------------------------------------------------------------------

def active_indices(bits: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(bits, dtype=bool))


def active_nodes(bits: np.ndarray) -> set[str]:
    g = graph()
    nodes: set[str] = set()
    for i in active_indices(bits):
        a, b = g.endpoints[i]
        nodes.add(a)
        nodes.add(b)
    return nodes


def _reachable_from_outfalls(active: set[int]) -> set[str]:
    """Undirected BFS over active edges, seeded from every outfall."""
    g = graph()
    active_adj: dict[str, list[int]] = defaultdict(list)
    for i in active:
        a, b = g.endpoints[i]
        active_adj[a].append(i)
        active_adj[b].append(i)
    seen: set[str] = set(g.outfalls)
    q = deque(g.outfalls)
    while q:
        node = q.popleft()
        for i in active_adj[node]:
            a, b = g.endpoints[i]
            nb = b if a == node else a
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def disconnected_inlets(bits: np.ndarray) -> list[str]:
    g = graph()
    reachable = _reachable_from_outfalls(set(active_indices(bits).tolist()))
    return [n for n in g.inlets if n not in reachable]


# --- repair ----------------------------------------------------------------------

def _edge_weight(i: int) -> float:
    g = graph()
    w = g.lengths[i]
    return w if g.downhill[i] else w * UPHILL_WEIGHT_FACTOR


def _shortest_path_edges(source: str, targets: set[str]) -> list[int] | None:
    """Dijkstra over the FULL candidate graph from ``source`` to the nearest node in
    ``targets``. Returns the list of gene indices on that path, or None if no path."""
    g = graph()
    dist = {source: 0.0}
    prev_edge: dict[str, int] = {}
    pq: list[tuple[float, str]] = [(0.0, source)]
    while pq:
        d, node = heapq.heappop(pq)
        if node in targets:
            # reconstruct
            edges = []
            cur = node
            while cur in prev_edge:
                e = prev_edge[cur]
                edges.append(e)
                a, b = g.endpoints[e]
                cur = b if a == cur else a
            return edges
        if d > dist.get(node, float("inf")):
            continue
        for i in g.node_adj[node]:
            a, b = g.endpoints[i]
            nb = b if a == node else a
            nd = d + _edge_weight(i)
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev_edge[nb] = i
                heapq.heappush(pq, (nd, nb))
    return None


def repair(bits: np.ndarray) -> np.ndarray:
    """Return a copy of ``bits`` in which every inlet can reach an outfall."""
    g = graph()
    bits = np.array(bits, dtype=np.int8, copy=True)
    if not g.outfalls or not g.inlets:
        return bits

    active = set(active_indices(bits).tolist())
    reachable = _reachable_from_outfalls(active)
    pending = [n for n in g.inlets if n not in reachable]

    guard = 0
    while pending and guard < len(g.inlets) + 5:
        guard += 1
        progressed = False
        for inlet in list(pending):
            if inlet in reachable:
                continue
            edges = _shortest_path_edges(inlet, reachable | set(g.outfalls))
            if not edges:
                continue  # inlet is unreachable even in the full graph -> data gap, skip
            for e in edges:
                bits[e] = 1
                active.add(e)
            progressed = True
        reachable = _reachable_from_outfalls(active)
        pending = [n for n in g.inlets if n not in reachable]
        if not progressed:
            break
    return bits


def random_chromosome(rng: np.random.Generator, p_on: float = 0.7) -> np.ndarray:
    """A random layout biased toward keeping conduits, then repaired to feasibility."""
    bits = (rng.random(n_genes()) < p_on).astype(np.int8)
    return repair(bits)
