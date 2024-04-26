#!/bin/bash
#SBATCH -o ./shells/output/cp_suscape-job.%j.out
#SBATCH --partition=titan
#SBATCH -J cp_suscape
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:4
#SBATCH --qos=titan

nvidia-smi

singularity run --nv ./shells/mmdet_0.5.sif bash ./tools/dist_train.sh ./configs/centerpoint/centerpoint_voxel01_suscape_3d.py 4 --work-dir ./work_dirs/centerpoint_voxel01_suscape_3d
