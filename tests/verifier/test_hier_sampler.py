import collections
from rl.hier_sampler import HierarchicalSampler


def make():
    tasks = ["swe"] * 40 + ["ut"] * 360
    variants = ["swe_32k"] * 40 + ["ut_function"] * 200 + ["ut_pytest"] * 100 + ["ut_stdio"] * 60
    return tasks, variants


def test_task_weights_respected_and_epoch_covers_len():
    t, v = make()
    s = HierarchicalSampler(t, v, task_weights={"swe": 0.3, "ut": 0.7}, seed=1)
    order = list(iter(s))
    assert len(order) == 400
    c = collections.Counter(t[i] for i in order)
    assert 0.25 < c["swe"] / 400 < 0.35            # ~30% swe even though only 10% of instances
    assert len(set(i for i in order if t[i] == "ut")) > 200   # broad coverage of the big bucket


def test_variant_weights_and_resume():
    t, v = make()
    s = HierarchicalSampler(t, v, task_weights={"ut": 1.0}, variant_weights={"ut_function": 0.5, "ut_pytest": 0.25, "ut_stdio": 0.25}, seed=3)
    it = iter(s)
    first = [next(it) for _ in range(100)]
    sd = s.state_dict()
    s2 = HierarchicalSampler(t, v, task_weights={"ut": 1.0}, variant_weights={"ut_function": 0.5, "ut_pytest": 0.25, "ut_stdio": 0.25}, seed=3)
    s2.load_state_dict(sd)
    rest2 = list(iter(s2))
    rest1 = list(it)
    assert rest1 == rest2 and len(first) + len(rest1) == 400
    c = collections.Counter(v[i] for i in first + rest1)
    assert c["ut_function"] > c["ut_pytest"] > 0 and c["swe_32k"] == 0


def test_default_proportional():
    t, v = make()
    s = HierarchicalSampler(t, v, seed=0)
    c = collections.Counter(t[i] for i in iter(s))
    assert 0.05 < c["swe"] / 400 < 0.16
