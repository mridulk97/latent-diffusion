#!/bin/bash
#SBATCH --account=imageomicswithanuj
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=1-00:00:00 # 10 minutes
#SBATCH --gres=gpu:3
#SBATCH --output=/home/mridul/latent-diffusion/slurm/dogs/slurm-%j-%x.out


module reset

module load Anaconda3/2020.11

source activate diff
module reset
which python

### Dogs Full VQGAN
# python main.py --name VQ_full --logdir /globalscratch/mridul/ldm/dogs/ --postfix _f4 --base configs/dogs/vq-gan-f4.yaml -t True --gpus 0,1,2,

# python main.py --name VQ_full --postfix _f4_day2 --logdir /globalscratch/mridul/ldm/dogs/model_runs/ --resume_from_checkpoint /globalscratch/mridul/ldm/dogs/model_runs/2024-04-26T01-21-57_VQ_full_f4/checkpoints/last.ckpt  -t True --gpus 0,1,2,
python main.py --logdir /globalscratch/mridul/ldm/dogs/model_runs/ --resume /globalscratch/mridul/ldm/dogs/model_runs/2024-04-26T01-21-57_VQ_full_f4 --base configs/dogs/vq-gan-f4.yaml -t True --gpus 0,1,2,


#### f4 CC
# python main.py --name Class_conditional --logdir /globalscratch/mridul/ldm/monkeys/ --base configs/monkeys/f4_class_conditional.yaml --postfix _days1_lr1e-6 -t True --gpus 0,

#### f4 T2T
# python main.py --name T2T --logdir /globalscratch/mridul/ldm/monkeys/ --base configs/monkeys/f4_t2t.yaml --postfix _days1_lr1e-6 -t True --gpus 0,


#### f4 HLE
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/dogs/ --base configs/dogs/dog_f4_level_encoding.yaml --postfix _custom_subset -t True --gpus 0,




exit;
