"""
Muhafiz.sol - Dataset A acquisition (Slither Audited Smart Contracts).

Why not ``datasets.load_dataset``?  The Hub repository ships a *loading script*
(``slither-audited-smart-contracts.py``). ``datasets`` >= 3.0 no longer executes
loading scripts, and the auto-converted parquet export on the Hub is marked
``partial``. We therefore read the raw shards the script itself reads
(``data/raw/contracts{0..8}.parquet``) together with ``big-splits.csv``,
``small-splits.csv`` and ``label_mappings.json``.

Keeping the raw Slither JSON (instead of the pre-collapsed integer labels) lets
us (a) see which contracts Slither *failed* on - the upstream script silently
labels those "safe" - and (b) build our own documented label map.

Only the byte ranges of the columns we need are downloaded (the bytecode column,
44% of every shard, is skipped: the app accepts source code only). Each shard is
reduced, appended to one zstd parquet file and deleted again, so the peak disk
footprint is one shard (~230 MB).

Run:  python -m src.ingest_slither
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.config import load_config, path

SIDE_FILES = ["big-splits.csv", "small-splits.csv", "label_mappings.json"]
NEEDED_COLUMNS = ["contracts", "source_code", "results"]
_PIECE = 8 << 20  # 8 MB range requests


def _get_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    """HTTP range request for bytes [start, end] (inclusive) with retries."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
            raise OSError(f"unexpected response {r.status_code} / {len(r.content)} bytes")
        except (requests.RequestException, OSError) as exc:
            print(f"      range {start}-{end} failed ({exc}); attempt {attempt}/{retries}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not fetch bytes {start}-{end} of {url}")


def fetch_parquet_columns(url: str, dest: Path, columns: List[str], workers: int = 4) -> Path:
    """Download only the byte ranges of ``columns`` from a remote parquet file.

    The bytecode column is ~44% of every shard and is never used, so instead of
    downloading whole files we read the parquet footer, locate the column chunks
    we need, and fetch just those ranges into a same-sized local file. Parquet
    readers seek by offset, so ``pq.read_table(dest, columns=columns)`` works.
    """
    if dest.exists():
        return dest
    tail_len = 1 << 16
    head = requests.get(url, headers={"Range": f"bytes=-{tail_len}"}, timeout=60)
    head.raise_for_status()
    total = int(head.headers["Content-Range"].split("/")[-1])
    tail = head.content
    footer_len = struct.unpack("<I", tail[-8:-4])[0]
    if tail[-4:] != b"PAR1":
        raise ValueError(f"{url} is not a parquet file")
    if footer_len + 8 > len(tail):
        tail = _get_range(url, total - footer_len - 8, total - 1)

    tmp = dest.with_suffix(".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        fh.truncate(total)
        fh.seek(total - len(tail))
        fh.write(tail)
    meta = pq.ParquetFile(tmp).metadata

    ranges: List[Tuple[int, int]] = []
    for rg in range(meta.num_row_groups):
        for ci in range(meta.num_columns):
            col = meta.row_group(rg).column(ci)
            if col.path_in_schema not in columns:
                continue
            start = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset:
                start = min(start, col.dictionary_page_offset)
            end = start + col.total_compressed_size - 1
            ranges += [(s, min(s + _PIECE - 1, end)) for s in range(start, end + 1, _PIECE)]

    def job(rng: Tuple[int, int]) -> Tuple[int, bytes]:
        return rng[0], _get_range(url, rng[0], rng[1])

    with open(tmp, "r+b") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        for offset, data in pool.map(job, ranges):
            fh.seek(offset)
            fh.write(data)
    tmp.replace(dest)
    return dest


def _download(url: str, dest: Path, retries: int = 4) -> Path:
    """Stream ``url`` to ``dest`` (skipped if it already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return dest
        except (requests.RequestException, OSError) as exc:
            print(f"      download failed ({exc}); attempt {attempt}/{retries}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download {url}")


def parse_slither(results_json: str) -> Dict[str, object]:
    """Extract success flag, error and the list of detector checks."""
    try:
        res = json.loads(results_json)
    except (TypeError, ValueError):
        return {"success": False, "error": "unparseable", "detectors": [], "impacts": []}
    body = res.get("results") or {}
    dets = body.get("detectors", []) if isinstance(body, dict) else []
    return {
        "success": bool(res.get("success", False)),
        "error": res.get("error"),
        "detectors": [d.get("check", "") for d in dets],
        "impacts": [f"{d.get('impact')}/{d.get('confidence')}" for d in dets],
    }


def ingest(force: bool = False) -> Path:
    """Download all shards and write ``data/interim/slither_contracts.parquet``."""
    cfg = load_config()["dataset"]
    raw_dir = path("slither_raw_dir")
    out = path("interim_dir") / "slither_contracts.parquet"
    if out.exists() and not force:
        print(f"[ingest] {out} exists - skipping (use force=True to rebuild)")
        return out

    base = cfg["slither_base_url"]
    for name in SIDE_FILES:
        _download(f"{base}/{name}", raw_dir / name)

    big = pd.read_csv(raw_dir / "big-splits.csv").assign(config="big")
    small = pd.read_csv(raw_dir / "small-splits.csv").assign(config="small")
    split_map = pd.concat([big, small]).drop_duplicates("contracts").set_index("contracts")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    schema = pa.schema([
        ("address", pa.string()), ("source_code", pa.large_string()),
        ("success", pa.bool_()), ("error", pa.string()),
        ("detectors", pa.list_(pa.string())), ("impacts", pa.list_(pa.string())),
        ("hf_config", pa.string()), ("hf_split", pa.string()), ("shard", pa.int16()),
    ])
    tmp_out = out.with_suffix(".parquet.part")
    try:
        writer = pq.ParquetWriter(tmp_out, schema, compression="zstd")
        for i in range(int(cfg["slither_shards"])):
            shard = raw_dir / f"contracts{i}.parquet"
            t0 = time.time()
            fetch_parquet_columns(f"{base}/raw/contracts{i}.parquet", shard, NEEDED_COLUMNS)
            df = pd.read_parquet(shard, columns=NEEDED_COLUMNS)
            parsed = [parse_slither(r) for r in df["results"]]
            rows: Dict[str, List] = {
                "address": ("0x" + df["contracts"]).tolist(),
                "source_code": df["source_code"].fillna("").tolist(),
                "success": [p["success"] for p in parsed],
                "error": [None if p["error"] is None else str(p["error"])[:500] for p in parsed],
                "detectors": [p["detectors"] for p in parsed],
                "impacts": [p["impacts"] for p in parsed],
                "hf_config": [split_map["config"].get(a) for a in df["contracts"]],
                "hf_split": [split_map["split"].get(a) for a in df["contracts"]],
                "shard": [i] * len(df),
            }
            writer.write_table(pa.Table.from_pydict(rows, schema=schema))
            print(f"[ingest] shard {i}: {len(df):,} rows in {time.time() - t0:.0f}s")
            del df, parsed, rows
            if not cfg.get("keep_raw_shards", False):
                shard.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.close()
    shutil.move(str(tmp_out), str(out))
    return out


if __name__ == "__main__":
    print(ingest())
