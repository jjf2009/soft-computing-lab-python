"""Phase 1 — Download the Bellinge dataset from the DTU public repository.

The dataset lives on DTU's Figshare instance (CC BY 4.0), published as the collection
with DOI 10.11583/DTU.c.5029124. A Figshare collection DOI of the form
``10.11583/DTU.c.<id>`` maps to collection ``<id>``; this script resolves the
collection, lists its articles and files, and streams the files we need into data/.

Streaming (chunked) download keeps memory flat even for the big time-series zips.

    python -m src.download_data            # download the needed files
    python -m src.download_data --list     # just print the manifest, download nothing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

from .common import find_inp, load_config, path

FIGSHARE_API = "https://api.figshare.com/v2"
CHUNK = 1 << 20  # 1 MiB


def collection_id_from_doi(doi: str) -> int:
    m = re.search(r"\.c\.(\d+)", doi)
    if not m:
        raise ValueError(f"Cannot extract a Figshare collection id from DOI {doi!r}")
    return int(m.group(1))


def get(url: str) -> list | dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def list_collection_files(collection_id: int) -> list[dict]:
    """Return a flat list of file records across every article in the collection."""
    files: list[dict] = []
    page = 1
    while True:
        articles = get(f"{FIGSHARE_API}/collections/{collection_id}/articles?page={page}&page_size=100")
        if not articles:
            break
        for art in articles:
            detail = get(f"{FIGSHARE_API}/articles/{art['id']}")
            for f in detail.get("files", []):
                files.append(
                    {
                        "article": detail.get("title", ""),
                        "name": f["name"],
                        "size": f.get("size", 0),
                        "download_url": f["download_url"],
                    }
                )
        page += 1
    return files


def wanted(name: str, cfg: dict, inp_only: bool = False) -> bool:
    name_l = name.lower()
    # The EPA SWMM model is distributed inside 7_SWMM.zip (extracted after download).
    is_model = name_l.endswith(".inp") or "swmm" in name_l
    small_meta = [w.lower() for w in cfg["dataset"]["files"] if not w.lower().endswith(".zip")]
    is_small_meta = name_l in small_meta
    if inp_only:
        # Colab demo: model + tiny metadata only; skip the heavy time-series zips.
        return is_model or is_small_meta
    if is_model:
        return True
    return any(name_l == w.lower() for w in cfg["dataset"]["files"])


def extract_swmm_zip(data_dir: Path) -> None:
    """Unzip the SWMM model archive so a bare .inp exists for the pipeline.

    Only the small SWMM zip is extracted; the big time-series zips are left packed
    and streamed later by preprocess.py.
    """
    import zipfile

    for z in data_dir.glob("*.zip"):
        if "swmm" not in z.name.lower():
            continue
        out = data_dir / (z.stem + "_extracted")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        print(f"  extracted {z.name} -> {out.name}/")


def stream_download(url: str, dest: Path, expected_size: int = 0) -> None:
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  skip (already present): {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
    tmp.rename(dest)
    print(f"  downloaded: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print manifest only, download nothing")
    ap.add_argument("--inp-only", action="store_true",
                    help="download just the SWMM model (for the Colab demo), skip the big zips")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = path("data_dir")
    cid = collection_id_from_doi(cfg["dataset"]["collection_doi"])
    print(f"Resolving Figshare collection {cid} ({cfg['dataset']['collection_doi']}) ...")

    files = list_collection_files(cid)
    print(f"Collection exposes {len(files)} files across its articles:\n")
    for f in files:
        flag = "GET" if wanted(f["name"], cfg, args.inp_only) else "   "
        print(f"  [{flag}] {f['name']:45s} {f['size']/1e6:8.1f} MB  <- {f['article'][:40]}")

    if args.list:
        return 0

    print("\nDownloading needed files ...")
    for f in files:
        if wanted(f["name"], cfg, args.inp_only):
            stream_download(f["download_url"], data_dir / f["name"], f["size"])

    extract_swmm_zip(data_dir)

    # Sanity check: confirm a SWMM model is present.
    try:
        inp = find_inp()
        print(f"\nSWMM model located: {inp.relative_to(data_dir.parent)}")
    except FileNotFoundError:
        print(
            "\nWARNING: no .inp found directly. The SWMM model may be inside a downloaded "
            "zip — unzip it into data/ and re-run `python -m src.preprocess`.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
