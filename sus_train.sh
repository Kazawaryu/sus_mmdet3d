#!/bin/bash
#SBATCH -o ./shells/output/cp_suscape-job.%j.out
#SBATCH --partition=titan
#SBATCH -J cp_suscape
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:4
#SBATCH --qos=titan

nvidia-smi

singularity run --nv ./shells/mmdet_0.8.sif bash ./tools/dist_train.sh ./configs/point_rcnn/point_rcnn_2x8_suscape-new.py 4 --work-dir ./work_dirs/point_rcnn_2x8_suscape-3d_online
