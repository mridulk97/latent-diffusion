#!/bin/bash
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=1-00:00:00 # 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mridul/latent-diffusion/slurm/monkeys/slurm-%j-%x.out


module reset

module load Anaconda3/2020.11

source activate diff
module reset
which python


#### f4 CC
# python main.py --name Class_conditional --logdir /globalscratch/mridul/ldm/monkeys/ --base configs/monkeys/f4_class_conditional.yaml --postfix _days1_lr1e-6 -t True --gpus 0,

#### f4 T2T
# python main.py --name T2T --logdir /globalscratch/mridul/ldm/monkeys/ --base configs/monkeys/f4_t2t.yaml --postfix _days1_lr1e-6 -t True --gpus 0,


#### f4 HLE
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/monkeys/ --base configs/monkeys/f4_level_encoding.yaml --postfix _days1_lr1e-6 -t True --gpus 0,

# V2
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/monkeys/model_runs/ --base configs/monkeys/f4_level_encoding_v2.yaml --postfix _days1_lr1e-6_v2 -t True --gpus 0,

## 6 levels
python main.py --name HLE --logdir  /globalscratch/mridul/ldm/monkeys/model_runs/ --base configs/monkeys/f4_level_encoding_6levels.yaml --postfix _days1_lr1e-6_6levels -t True --gpus 0,



exit;
