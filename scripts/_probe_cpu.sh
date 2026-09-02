#!/bin/bash
#SBATCH -p koni -w gpu48 -c 4 --mem=16G -t 00:03:00 -J cpuprobe
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/cpuprobe_%j.out
hostname; echo CVD=$CUDA_VISIBLE_DEVICES; nproc; singularity --version; id; cat /etc/subuid 2>/dev/null | grep -c $USER; singularity build --help 2>&1 | grep -iE "fakeroot|remote" | head -3; podman info 2>&1 | head -5
