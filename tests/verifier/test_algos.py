import numpy as np, torch, pytest
verl = pytest.importorskip("verl")
from rl.algos import compute_grpo_ddca_advantage


def test_ddca_shorter_correct_gets_higher_advantage():
    # one group of 4: two correct (lengths 100 vs 300), two wrong
    rewards = torch.zeros(4, 300)
    rewards[0, 99] = 1.0; rewards[1, 299] = 1.0
    mask = torch.zeros(4, 300); mask[0, :100] = 1; mask[1, :300] = 1; mask[2, :50] = 1; mask[3, :50] = 1
    idx = np.array(["a"] * 4)
    adv, _ = compute_grpo_ddca_advantage(rewards, mask, idx, answer_match=[1, 1, 0, 0], config={"ddca_beta": 0.3})
    a = adv.sum(-1) / mask.sum(-1)
    assert a[0] > a[1] > 0 > a[2] == a[3]
    adv0, _ = compute_grpo_ddca_advantage(rewards, mask, idx, answer_match=[1, 1, 0, 0], config={"ddca_beta": 0.0})
    a0 = adv0.sum(-1) / mask.sum(-1)
    assert torch.isclose(a0[0], a0[1])  # beta=0 -> plain GRPO
