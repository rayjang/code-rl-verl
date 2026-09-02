"""Debug helper: for one instance per repo in a list, (1) show where the package is imported from
inside the image, (2) run the gold patch with logs kept and print the pytest tail."""
import json, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from verifier.registry import Registry
from verifier.swe_runner import SweSmithRunner, SING
reg = Registry.get()
recs = [json.loads(l) for l in open(f"{ROOT}/environments/manifests/validation_swe.jsonl")]
want = {}
for r in recs:
    st = r.get("status")
    g = r.get("gold", {})
    key = r["repo"]
    if g.get("ran") and g.get("n_f2p", 0) < g.get("t_f2p", 0) and key not in want:
        want[key] = r["instance_id"]
runner = SweSmithRunner(f"{ROOT}/environments/sif", os.environ.get("VERIFIER_RUN_DIR", "/tmp/dbg_run"), f"{ROOT}/environments/cache", keep_workdir=True)
for repo, iid in want.items():
    inst = reg.swe[iid]; sif = runner.sif_for(inst)
    pkg = repo.split("/")[-1].split("__")[1].split(".")[0].replace("-", "_")
    probe = (f"source /opt/miniconda3/bin/activate testbed 2>/dev/null; cd /testbed; python -c 'import {pkg}; print(\"IMPORT\", {pkg}.__file__)' 2>&1 | tail -1; "
             f"pip show {pkg} 2>/dev/null | grep -iE 'Location|Editable'; ls /testbed | head -30 | tr '\\n' ' '; echo; git -c safe.directory='*' status --porcelain 2>/dev/null | head -5; ls /testbed/src 2>/dev/null | head; ")
    p = subprocess.run([SING, "exec", "--containall", "--env", "HOME=/tmp", sif, "bash", "-c", probe], capture_output=True, text=True, timeout=300)
    print("=" * 100); print(repo, iid); print(p.stdout[-1500:], p.stderr[-300:])
    er = runner.run(inst, inst["gold_patch"])
    print("gold:", er.ran, er.err_kind.value, "f2p", er.f2p.n_passed, "/", er.f2p.total, "missing", len(er.f2p.missing), "p2p", er.p2p.n_passed, "/", er.p2p.total)
    print("F2P failed ids:", list(er.f2p.failed)[:5], "missing:", list(er.f2p.missing)[:5])
    print("LOG TAIL:\n", er.log_tail[-2500:])
