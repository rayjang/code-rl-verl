"""Instance registry: loads the immutable index files (SWE-smith + unit-test tracks), retargets the
`sif` paths to the local image directory, and exposes lookup by instance_id.

Index location: $RL_INDEX_DIR (default: sources/rl_code_v1/data/index). Nothing is written back.
"""
from __future__ import annotations

import glob
import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(ROOT, "sources", "rl_code_v1", "data", "index")
DEFAULT_SIF_DIR = os.path.join(ROOT, "environments", "sif")


class Registry:
    _inst = None
    _lock = threading.Lock()

    def __init__(self, index_dir: str | None = None, sif_dir: str | None = None):
        self.index_dir = index_dir or os.environ.get("RL_INDEX_DIR", DEFAULT_INDEX)
        self.sif_dir = sif_dir or os.environ.get("RL_SIF_DIR", DEFAULT_SIF_DIR)
        self.swe: dict = {}
        self.swe_meta: dict = {}
        self.ut: dict = {}
        self.ut_meta: dict = {}
        self._load()

    @classmethod
    def get(cls) -> "Registry":
        with cls._lock:
            if cls._inst is None:
                cls._inst = Registry()
            return cls._inst

    def _read(self, pattern):
        for p in sorted(glob.glob(os.path.join(self.index_dir, pattern))):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def _load(self):
        for d in self._read("t15_code_swesmith*.full.jsonl"):
            self.swe[d["instance_id"]] = d
        for d in self._read("t15_code_swesmith*.learnable.jsonl"):
            d["sif"] = os.path.join(self.sif_dir, d["image_name"].rsplit("/", 1)[-1].replace(":", "_") + ".sif")
            self.swe_meta[d["instance_id"]] = d
        for d in self._read("t15_code_unittest*.full.jsonl"):
            self.ut[d["instance_id"]] = d
        for d in self._read("t15_code_unittest*.learnable.jsonl"):
            self.ut_meta[d["instance_id"]] = d
        if not self.swe and not self.ut:
            raise RuntimeError(f"no index files found under {self.index_dir} (set RL_INDEX_DIR)")

    def track_of(self, instance_id: str, data_source: str = "") -> str:
        if instance_id in self.ut or str(data_source).startswith("t15_unittest"):
            return "unittest"
        if instance_id in self.swe or str(data_source).startswith("t15_repo_patch"):
            return "swe"
        return "unknown"

    def instance(self, instance_id: str) -> dict | None:
        return self.swe.get(instance_id) or self.ut.get(instance_id)

    def meta(self, instance_id: str) -> dict | None:
        return self.swe_meta.get(instance_id) or self.ut_meta.get(instance_id)

    def buggy_files(self, instance_id: str) -> dict | None:
        d = self.swe.get(instance_id)
        if not d:
            return None
        bf = d.get("buggy_files")
        if isinstance(bf, str):
            try:
                return json.loads(bf)
            except Exception:
                return None
        return bf
