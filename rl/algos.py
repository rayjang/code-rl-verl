"""Custom verl algorithm pieces, registered on import (set `actor_rollout_ref.model.external_lib=rl.algos`
so every worker process imports this module; the trainer process imports it via rl.main_ppo_code).

1. `grpo_ddca` advantage estimator
   GRPO outcome advantage (optionally std-normalised, "std on") minus beta * DDCA length advantage
   (Peng et al. 2026, "Think Dense, Not Long: Dynamic Decoupled Conditional Advantage", arXiv 2602.02099):
     C = {i : answer_match_i == 1}, n = |C|, N = group size
     z_i = (len_i - mean_C len) / std_C len,  r_i^len = sigmoid(z_i)
     A_i^len = (n/N) * (r_i^len - mean_{j in C, j != i} r_j^len)   for i in C, else 0
     A_i = A_i^acc - beta * A_i^len
   The correct cluster comes from the reward dict key `answer_match` (pure 0/1 = resolved), exactly the role
   the local scorers document ("DDCA builds the correct cluster from answer_match").
   Config: algorithm.ddca_beta (default 0.3), algorithm.ddca_min_cluster (default 2).

2. `gspo_tokenmean` policy loss = verl's GSPO sequence-level ratio + clipping, but aggregated with the
   configured `loss_agg_mode` (token-mean) instead of the hard-coded seq-mean-token-mean.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss
import verl.utils.torch_functional as verl_F


def _cfg_get(config, key, default):
    if config is None:
        return default
    try:
        v = config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)
    except Exception:
        v = default
    return default if v is None else v


@register_adv_est("grpo_ddca")
def compute_grpo_ddca_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True, config=None, **kwargs):
    scores = token_level_rewards.sum(dim=-1)
    lengths = response_mask.sum(dim=-1).float()
    answer_match = kwargs.get("answer_match")
    beta = float(_cfg_get(config, "ddca_beta", 0.3))
    min_cluster = int(_cfg_get(config, "ddca_min_cluster", 2))
    if answer_match is None:
        correct = (scores >= 0.999).float()
    else:
        correct = torch.as_tensor(np.asarray(answer_match, dtype=np.float32), device=scores.device)
    bsz = scores.shape[0]
    groups = defaultdict(list)
    for i in range(bsz):
        groups[index[i]].append(i)
    adv = torch.zeros_like(scores)
    with torch.no_grad():
        for idxs in groups.values():
            g = torch.tensor(idxs, device=scores.device)
            s = scores[g]
            if len(idxs) == 1:
                a_acc = torch.zeros_like(s)
            else:
                mean, std = s.mean(), s.std()
                a_acc = (s - mean) / (std + epsilon) if norm_adv_by_std_in_grpo else (s - mean)
            a_len = torch.zeros_like(s)
            c_mask = correct[g] > 0.5
            n = int(c_mask.sum().item())
            if beta > 0 and n >= min_cluster:
                L = lengths[g][c_mask]
                mu, sd = L.mean(), L.std()
                z = (L - mu) / (sd + epsilon) if sd > 0 else torch.zeros_like(L)
                r_len = torch.sigmoid(z)
                tot = r_len.sum()
                rloo = (tot - r_len) / max(1, n - 1)
                a_len[c_mask] = (n / len(idxs)) * (r_len - rloo)
            adv[g] = a_acc - beta * a_len
    advantages = adv.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_policy_loss("gspo_tokenmean")
def compute_policy_loss_gspo_tokenmean(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean",
                                       config=None, rollout_is_weights=None):
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, {"actor/pg_clipfrac": pg_clipfrac.detach().item(), "actor/ppo_kl": ppo_kl.detach().item(),
                     "actor/pg_clipfrac_lower": 0.0}
