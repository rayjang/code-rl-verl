"""Custom verl entrypoint = verl.trainer.main_ppo with two additions, applied INSIDE the Ray TaskRunner
actor (a driver-side monkeypatch would not reach the actor process):
  1. hierarchical sampler (data.hier_sampler present in config)  -> patches create_rl_sampler
  2. import of rl.rusca_agent_loop so the agent loop registry knows `rusca_scaffold_agent`
Everything else is stock verl 0.9.0 (TaskRunnerV1.run copied verbatim).

Usage: python -m rl.main_ppo_code <hydra overrides>
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hydra  # noqa: E402
import ray  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from pprint import pprint  # noqa: E402

import verl.trainer.main_ppo as main_ppo  # noqa: E402
from verl.trainer.ppo.utils import need_critic, need_reference_policy  # noqa: E402
from verl.utils.config import validate_config  # noqa: E402
from verl.utils.device import auto_set_device  # noqa: E402
from verl.utils.logging_utils import configure_verl_logging  # noqa: E402


def _install_patches(config):
    import verl.trainer.ppo.v1.trainer_base as v1_base
    from verl.trainer.ppo.utils import create_rl_sampler as orig
    hier = config.data.get("hier_sampler", None)
    if hier is not None:
        cfg = OmegaConf.to_container(hier, resolve=True)

        def patched(data_config, dataset):
            from rl.hier_sampler import build_from_dataset
            s = build_from_dataset(dataset, cfg, seed=int(data_config.get("seed") or 0))
            print(f"[hier_sampler] tasks={s.task_list} variants={s.variants_of} weights={cfg.get('task_weights')}/"
                  f"{cfg.get('variant_weights')} n={len(s)}", flush=True)
            return s
        v1_base.create_rl_sampler = patched
    import rl.rusca_agent_loop  # noqa: F401  (registers the agent loop in this process)
    import rl.algos  # noqa: F401  (registers grpo_ddca / gspo_tokenmean in the trainer process)


@ray.remote
class CodeTaskRunner(main_ppo.TaskRunnerV1.__ray_metadata__.modified_class):
    def run(self, config: DictConfig):
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        _install_patches(config)
        return super().run(config)


@hydra.main(config_path=os.path.join(os.path.dirname(main_ppo.__file__), "config"), config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    assert config.trainer.use_v1, "this entrypoint targets the v1 trainer"
    main_ppo.run_ppo(config, task_runner_class=CodeTaskRunner)


if __name__ == "__main__":
    main()
