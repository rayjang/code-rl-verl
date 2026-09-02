"""Second debug pass: print the HEAD of the pytest output for repos with setup errors, and the git apply
error for inflect instances whose gold still fails to apply."""
import json, os, sys, subprocess, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from verifier.registry import Registry
from verifier.swe_runner import SweSmithRunner
from verifier.patch_apply import apply_patch
reg = Registry.get()
recs = [json.loads(l) for l in open(f"{ROOT}/environments/manifests/validation_swe.jsonl")]
runner = SweSmithRunner(f"{ROOT}/environments/sif", os.environ.get("VERIFIER_RUN_DIR", "/tmp/dbg_run"), f"{ROOT}/environments/cache_v4", keep_workdir=True)
targets = {}
for r in recs:
    key = r["repo"].split("/")[-1]
    if key.startswith(("joke2k", "Cog-Creators")) and r.get("gold", {}).get("ran") and not r["gold"]["resolved"] and key not in targets:
        targets[key] = r["instance_id"]
    if key.startswith("jaraco__inflect") and r.get("gold", {}).get("apply") == "patch_apply_fail" and key not in targets:
        targets[key] = r["instance_id"]
for key, iid in targets.items():
    inst = reg.swe[iid]
    print("=" * 90); print(key, iid, "F2P:", inst["FAIL_TO_PASS"][:3])
    if key.startswith("jaraco"):
        tar, err = runner.base_tar(inst); wd = tempfile.mkdtemp(prefix="dbg."); subprocess.run(["tar", "-xf", tar, "-C", wd], check=True)
        tree = os.path.join(wd, "tb")
        r1 = apply_patch(inst["gold_patch"], tree); print("strict:", r1.status.value, r1.stderr_tail[-400:])
        r2 = apply_patch(inst["gold_patch"], tree, ignore_whitespace=True); print("ignore-ws:", r2.status.value, r2.stderr_tail[-400:])
        p = subprocess.run(["git", "apply", "--check", "-R", "-"], cwd=tree, input=inst["gold_patch"].encode(), capture_output=True); print("reverse rc:", p.returncode)
        print("gold head:", inst["gold_patch"][:600])
        continue
    er = runner.run(inst, inst["gold_patch"])
    print("gold:", er.ran, er.err_kind.value, "f2p", er.f2p.n_passed, "/", er.f2p.total, "p2p", er.p2p.n_passed, "/", er.p2p.total)
    # find the work dir (kept) and print head of out.log
    wds = sorted([d for d in os.listdir(runner.run_dir) if d.startswith(iid[:40])], key=lambda d: os.path.getmtime(os.path.join(runner.run_dir, d)))
    if wds:
        out = open(os.path.join(runner.run_dir, wds[-1], "out.log"), errors="replace").read()
        print("--- OUT HEAD ---"); print(out[:3500]); print("--- OUT ERRORS section ---")
        i = out.find("= ERRORS ="); print(out[i:i + 3000] if i >= 0 else "(no ERRORS section)")
