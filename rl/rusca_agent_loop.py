"""RUSCA scaffold agent loop for verl 0.9 (registered as `rusca_scaffold_agent`).

Behaviour (reconstructed; see docs/rusca_reverse_engineering.md):
  * the rollout prompt = original chat + scaffold criteria (selected by rank, count from lam(step))
    appended to the last user turn;
  * the returned AgentLoopOutput carries the UN-scaffolded prompt ids so the actor/ref compute
    log-probs on the training prompt (hence calculate_log_probs must be False: the rollout log-probs
    are conditioned on the scaffolded prompt and must not be used as old log-probs);
  * the number of injected criteria is written into reward_extra_info so it appears in metrics;
  * the current global step is written to $RUSCA_STEP_FILE for the reward workers (rusca_stage).
Config: actor_rollout_ref.rollout.custom.rusca.{enable,total_steps,center_frac,steepness,group_jitter,max_criteria,profile}
"""
from __future__ import annotations

import os
import sys
from typing import Any
from uuid import uuid4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register  # noqa: E402
from verl.utils.profiler import simple_timer  # noqa: E402

from verifier.rubric import parse_rubrics  # noqa: E402
from verifier.rusca import RuscaConfig, inject, n_inject, select_criteria, stage  # noqa: E402


def _rusca_cfg(config) -> RuscaConfig:
    try:
        custom = config.actor_rollout_ref.rollout.get("custom", {}) or {}
        d = dict(custom.get("rusca", {}) or {})
    except Exception:
        d = {}
    if "stage_boundaries" in d:
        d["stage_boundaries"] = tuple(d["stage_boundaries"])
    return RuscaConfig(**{k: v for k, v in d.items() if k in RuscaConfig.__dataclass_fields__})


@register("rusca_scaffold_agent")
class RuscaScaffoldAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rusca = _rusca_cfg(self.config)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.step_file = os.environ.get("RUSCA_STEP_FILE", "")
        self._last_step_written = None

    def _write_step(self, step: int):
        if self.step_file and step != self._last_step_written:
            try:
                tmp = self.step_file + ".tmp"
                with open(tmp, "w") as f:
                    f.write(str(step))
                os.replace(tmp, self.step_file)
                self._last_step_written = step
            except Exception:
                pass

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        ei = kwargs.get("extra_info") or {}
        if hasattr(ei, "keys"):
            ei = dict(ei)
        step = int(kwargs.get("global_steps", ei.get("global_step", 0)) or 0)
        validate = bool(kwargs.get("validate", False))
        self._write_step(step)

        # --- scaffold selection -------------------------------------------------------------
        rubrics = parse_rubrics(ei.get("rubrics")) if self.rusca.enable and not validate else []
        variant_key = f"{ei.get('instance_id', kwargs.get('index', ''))}:{priority}"
        k = n_inject(step, len(rubrics), self.rusca, variant_key) if rubrics else 0
        criteria = select_criteria(rubrics, k) if k else []
        lang = str(ei.get("lang", "en"))
        rollout_messages = inject(messages, criteria, lang) if criteria else messages

        # --- tokenise both prompts ------------------------------------------------------------
        train_prompt_ids = await self.apply_chat_template(messages)
        rollout_prompt_ids = await self.apply_chat_template(rollout_messages) if criteria else train_prompt_ids

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=rollout_prompt_ids,
                                                        sampling_params=sampling_params, priority=priority)
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_ids = output.token_ids[: self.response_length]
        out = AgentLoopOutput(
            prompt_ids=train_prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,                       # scaffold-conditioned; never reuse as old log-probs
            num_turns=2,
            metrics=metrics,
            extra_fields={"reward_extra_info": {"rusca_n_inject": float(k), "rusca_step": float(step),
                                                "rusca_prompt_delta_tokens": float(len(rollout_prompt_ids) - len(train_prompt_ids))},
                          # merged into extra_info by verl's reward managers -> overlong shaping + RUSCA stage in the reward
                          "tool_extra_fields": {"valid_response_length": len(response_ids), "max_response_length": self.response_length,
                                                "global_step": step, "rusca_n_inject": k}},
        )
        return out
