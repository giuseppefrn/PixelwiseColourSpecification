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
python3 autoencoder.py --epochs 150 --split_train 1 --add_noise 1 --noise_std 0.1 --alpha 10 --data_dir /scratch/gfurnari/datasets/zoomed/D65 --label_dir /scratch/gfurnari/datasets/zoomed/SHADE --output_dir /scratch/gfurnari/outputs/autoencoder/ --experiment_name "autoencoder model - mse loss - gradient reg"