"""Hierarchical (task -> variant -> instance) sampler for verl's StatefulDataLoader.

Semantics reconstructed from the project notes ("데이터 task별 sampling과 task 내 variant sampling"):
level 1 picks a task (extra_info.task_tag / data_source) with configurable weights, level 2 picks a
variant inside the task (extra_info.variant_type) with configurable weights, level 3 picks an
instance uniformly WITHOUT replacement within the (task, variant) bucket for the current epoch.
Exhausted buckets are refilled (reshuffled) so that len(sampler) == len(dataset) per epoch and rare
buckets are re-visited instead of silently dropped. Optional per-instance weights (e.g. difficulty
band up-weighting) reshape the within-bucket order.

State (epoch, cursor, rng) is checkpointable for resumption (StatefulDataLoader calls state_dict()).
"""
from __future__ import annotations

import collections
import random
from typing import Iterator, Optional, Sequence

from torch.utils.data import Sampler


class HierarchicalSampler(Sampler[int]):
    def __init__(self, tasks: Sequence[str], variants: Sequence[str], *, task_weights: Optional[dict] = None,
                 variant_weights: Optional[dict] = None, instance_weights: Optional[Sequence[float]] = None,
                 seed: int = 0, num_samples: Optional[int] = None):
        assert len(tasks) == len(variants)
        self.tasks, self.variants = list(tasks), list(variants)
        self.n = len(tasks)
        self.num_samples = num_samples or self.n
        self.task_weights = dict(task_weights or {})
        self.variant_weights = dict(variant_weights or {})
        self.instance_weights = list(instance_weights) if instance_weights is not None else None
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.buckets: dict = collections.defaultdict(list)
        for i, (t, v) in enumerate(zip(self.tasks, self.variants)):
            self.buckets[(t, v)].append(i)
        self.task_list = sorted({t for t, _ in self.buckets})
        self.variants_of = {t: sorted({v for tt, v in self.buckets if tt == t}) for t in self.task_list}

    # ------------------------------------------------------------------ weights
    def _tw(self, t):
        return float(self.task_weights.get(t, len([i for (tt, _) in self.buckets for i in [0] if tt == t]) or 1.0))

    def _task_probs(self):
        if self.task_weights:
            w = [float(self.task_weights.get(t, 0.0)) for t in self.task_list]
        else:  # proportional to bucket sizes (== uniform over instances)
            w = [sum(len(self.buckets[(t, v)]) for v in self.variants_of[t]) for t in self.task_list]
        s = sum(w) or 1.0
        return [x / s for x in w]

    def _variant_probs(self, t):
        vs = self.variants_of[t]
        if any(v in self.variant_weights for v in vs):
            w = [float(self.variant_weights.get(v, 0.0)) for v in vs]
        else:
            w = [len(self.buckets[(t, v)]) for v in vs]
        s = sum(w) or 1.0
        return vs, [x / s for x in w]

    # ------------------------------------------------------------------ order
    def _epoch_order(self, epoch: int) -> list:
        rng = random.Random(self.seed * 1000003 + epoch)
        queues = {}
        for key, idxs in self.buckets.items():
            order = list(idxs)
            if self.instance_weights:
                # weighted shuffle (Efraimidis-Spirakis): key = u^(1/w)
                order.sort(key=lambda i: -(rng.random() ** (1.0 / max(1e-6, self.instance_weights[i]))))
            else:
                rng.shuffle(order)
            queues[key] = collections.deque(order)
        out = []
        tp = self._task_probs()
        while len(out) < self.num_samples:
            t = rng.choices(self.task_list, tp)[0]
            vs, vp = self._variant_probs(t)
            v = rng.choices(vs, vp)[0]
            q = queues[(t, v)]
            if not q:                           # bucket exhausted this epoch -> refill (reshuffled)
                order = list(self.buckets[(t, v)]); rng.shuffle(order); q.extend(order)
            out.append(q.popleft())
        return out

    def __iter__(self) -> Iterator[int]:
        order = self._epoch_order(self.epoch)
        start = self.cursor
        for i in range(start, len(order)):
            self.cursor = i + 1
            yield order[i]
        self.epoch += 1
        self.cursor = 0

    def __len__(self):
        return self.num_samples

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed}

    def load_state_dict(self, sd):
        self.epoch, self.cursor, self.seed = int(sd["epoch"]), int(sd["cursor"]), int(sd.get("seed", self.seed))


def build_from_dataset(dataset, cfg: dict, seed: int = 0) -> HierarchicalSampler:
    """cfg: {task_key, variant_key, task_weights, variant_weights, weight_key}. Reads extra_info columns
    from a verl RLHFDataset (dataset.dataframe is a HF Dataset with an extra_info column)."""
    task_key = cfg.get("task_key", "task_tag")
    variant_key = cfg.get("variant_key", "variant_type")
    weight_key = cfg.get("weight_key")
    df = dataset.dataframe
    eis = df["extra_info"] if "extra_info" in df.column_names else [{} for _ in range(len(df))]
    tasks = [str((e or {}).get(task_key, "")) for e in eis]
    variants = [str((e or {}).get(variant_key, "")) for e in eis]
    iw = [float((e or {}).get(weight_key, 1.0) or 1.0) for e in eis] if weight_key else None
    return HierarchicalSampler(tasks, variants, task_weights=cfg.get("task_weights"), variant_weights=cfg.get("variant_weights"),
                               instance_weights=iw, seed=seed)
