#!/bin/bash
#SBATCH -p koni -w gpu48 -c 4 --mem=16G -t 00:30:00 -J ut_sandbox
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/ut_sandbox_%j.out
# Unit-test sandbox: python:3.11-slim-bookworm .sif + a venv living on the host (bind-mounted).
# (no fakeroot on this cluster -> cannot run %post of unittest.def; this reproduces its content)
set -euo pipefail
ROOT=/scratch/r919a03/code_verl_test
SIF=$ROOT/environments/sif/python_3.11-slim-bookworm.sif
VENV=$ROOT/environments/ut_venv
mkdir -p $VENV
singularity exec --containall --bind $VENV:/ut_venv $SIF python3 -m venv /ut_venv
singularity exec --containall --bind $VENV:/ut_venv $SIF /ut_venv/bin/pip install --no-cache-dir pytest==8.3.4 numpy==2.2.1 sympy==1.13.3
# test in isolated mode (as the verifier will run it)
mkdir -p /tmp/ut_probe && echo 'def test_a(): assert 1+1==2' > /tmp/ut_probe/test_probe.py
singularity exec --containall --no-home --cleanenv --net --network=none --writable-tmpfs \
  --bind $VENV:/ut_venv:ro --bind /tmp/ut_probe:/work --pwd /work \
  --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,PYTHONDONTWRITEBYTECODE=1,PYTHONHASHSEED=0 \
  $SIF /ut_venv/bin/python -m pytest -q -p no:cacheprovider /work
singularity exec --containall --net --network=none --bind $VENV:/ut_venv:ro $SIF /ut_venv/bin/python -c "import numpy, sympy, pytest; print('deps ok', numpy.__version__, sympy.__version__, pytest.__version__)"
singularity exec --containall --net --network=none --bind $VENV:/ut_venv:ro $SIF /ut_venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)" && echo "NETWORK NOT BLOCKED (BAD)" || echo "network blocked OK"
echo UT_SANDBOX_DONE
