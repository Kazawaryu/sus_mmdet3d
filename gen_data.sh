#!/bin/bash
#SBATCH -o ./shells/output/gen-job.%j.out
#SBATCH --partition=titan
#SBATCH -J gen_data
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --qos=titan

nvidia-smi

singularity run --nv ./shells/mmdet_0.5.sif python tools/create_data.py suscape --root-path ~/dataset/online_s1s2/ --out-dir data/suscape-online --extra-tag suscape --version v1.0-trainval
