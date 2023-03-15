#!/bin/bash -e
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/siamese-%j.out
#SBATCH --error=logs/siamese-%j.err
#SBATCH --mail-user=giuseppe.furnari@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

. pytorch-venv/bin/activate
cd color-segm
python3 siamese.py --epochs 50 --data_dir /scratch/gfurnari/datasets/zoomed/D65 --label_dir /scratch/gfurnari/datasets/zoomed/SHADE --output_dir /scratch/gfurnari/outputs --experiment_name siamese-zoomed-sigmoidGEN