from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import BenchmarkEntry


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_benchmark(path: Path) -> List[BenchmarkEntry]:
    obj = load_json(path)
    if isinstance(obj, dict):
        entries = obj.get("entries", [])
    elif isinstance(obj, list):
        entries = obj
    else:
        raise TypeError(f"Unsupported benchmark root type: {type(obj)}")

    out: List[BenchmarkEntry] = []
    root = path.parent
    for raw in entries:
        entry = BenchmarkEntry.from_dict(raw)
        entry.resolve_paths(root)
        out.append(entry)
    return out


def sanitize_slug(text: str, default: str = "unknown") -> str:
    text = (text or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def parse_iter_from_checkpoint(path: str) -> int:
    m = re.search(r"chkpnt(\d+)\.pth$", str(path))
    return int(m.group(1)) if m else -1


def parse_iter_from_name(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else -1


def find_latest_ours_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.glob("ours_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: parse_iter_from_name(p.name))
    return candidates[-1]


def find_latest_render_dir(model_dir: Path, split: str = "train") -> Optional[Path]:
    split_dir = model_dir / split
    latest = find_latest_ours_dir(split_dir)
    if latest is None:
        return None
    render_dir = latest / "renders"
    if render_dir.exists():
        return render_dir
    return None


def find_latest_iteration(model_dir: Path) -> int:
    pc_root = model_dir / "point_cloud"
    if not pc_root.exists():
        return -1
    candidates = [p for p in pc_root.glob("iteration_*") if p.is_dir()]
    if not candidates:
        return -1
    iters = []
    for c in candidates:
        try:
            iters.append(int(c.name.split("_")[-1]))
        except Exception:
            continue
    return max(iters) if iters else -1


def sorted_pngs(path: Path) -> List[Path]:
    return sorted(path.glob("*.png"), key=lambda p: p.name)


def intersect_filenames(a: Sequence[Path], b: Sequence[Path]) -> List[str]:
    sa = {p.name for p in a}
    sb = {p.name for p in b}
    return sorted(sa & sb)


def maybe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = load_json(path)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}


def get_git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    if n <= 0:
        yield seq
        return
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        rows.append({k: v for k, v in zip(header, cols)})
    return header, rows
