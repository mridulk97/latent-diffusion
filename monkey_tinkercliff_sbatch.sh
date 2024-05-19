#!/bin/bash
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=1-00:00:00 # 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mridul/latent-diffusion/slurm-%j-%x.out

module reset

module load Anaconda3/2020.11

source activate ldm
module reset
which python

# # F4 Monkey
# python main.py --name Monkey_VQ --logdir /fastscratch/mridul/new_diffusion_models/ldm/monkey_vq-base --postfix _f4_org --base configs/monkeys/vq-gan-f4.yaml -t True --gpus 0,1,2,

## Class conditional
python main.py --name Monkey_ --logdir /fastscratch/mridul/new_diffusion_models/ldm/monkey_vq-base --postfix _f4_cc --base configs/monkeys/f4_class_conditional.yaml -t True --gpus 0,


exit;
