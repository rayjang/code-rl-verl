import json, os, sys, subprocess, tempfile, shlex
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from verifier.registry import Registry
from verifier.swe_runner import SweSmithRunner, SING
from verifier.patch_apply import apply_patch
reg = Registry.get()
recs = [json.loads(l) for l in open(f"{ROOT}/environments/manifests/validation_swe.jsonl")]
runner = SweSmithRunner(f"{ROOT}/environments/sif", os.environ.get("VERIFIER_RUN_DIR", "/tmp/dbg_run"), f"{ROOT}/environments/cache_v4")
for repo in ("joke2k__faker", "Cog-Creators__Red-DiscordBot", "pyupio__safety", "agronholm__typeguard"):
    r = next(x for x in recs if repo in x["repo"] and x.get("gold", {}).get("ran") and not x["gold"]["resolved"])
    inst = reg.swe[r["instance_id"]]; sif = runner.sif_for(inst)
    tar, err = runner.base_tar(inst); wd = tempfile.mkdtemp(prefix="dbg3."); subprocess.run(["tar", "-xf", tar, "-C", wd], check=True)
    tree = os.path.join(wd, "tb"); ap = apply_patch(inst["gold_patch"], tree, ignore_whitespace=True)
    f2p, p2p = runner.test_ids(inst)
    tests = " ".join(shlex.quote(t) for t in f2p[:3] + p2p[:2])
    script = ("source /opt/miniconda3/bin/activate testbed 2>/dev/null || export PATH=/opt/miniconda3/envs/testbed/bin:$PATH\n"
              "cd /testbed; echo PY=$(which python); pip list 2>/dev/null | grep -iE 'pytest|asyncio|faker|freezegun|validators|randomly|xdist' | tr '\\n' ';'; echo\n"
              "echo '--- env vars ---'; env | grep -iE 'PYTEST|HOME|USER|LANG' | tr '\\n' ';'; echo\n"
              f"python -m pytest -rA --tb=short --color=no -p no:cacheprovider -q {tests} 2>&1 | head -120\n")
    open(f"{wd}/dbg.sh", "w").write(script)
    for iso_label, iso in (("isolated", ["--containall", "--no-home", "--cleanenv", "--net", "--network=none"]), ("plain", [])):
        p = subprocess.run([SING, "exec", *iso, "--writable-tmpfs", "--bind", f"{tree}:/testbed", "--bind", f"{wd}:/wd", "--pwd", "/testbed", sif, "bash", "/wd/dbg.sh"],
                           capture_output=True, text=True, timeout=600)
        print("=" * 30, repo, r["instance_id"], iso_label, "apply:", ap.status.value); print((p.stdout + p.stderr)[:6000])
    subprocess.run(["rm", "-rf", wd])
