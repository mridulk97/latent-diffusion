#!/bin/bash
#SBATCH --account=ece6524-spring2024
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=0-20:00:00 # 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mridul/latent-diffusion/slurm/multilevel/slurm-%j-%x.out


module reset

module load Anaconda3/2020.11

source activate diff
module reset
which python


#### f4 CC
# python main.py --name Class_conditional --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/lr1e-6/ --base configs/fishes/fishes_f4_class_conditional.yaml --postfix _days2_lr1e-6 -t True --gpus 0,

# channels 32
# python main.py --name Class_conditional --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/ --base configs/fishes/fishes_f4_class_conditional_channels32.yaml --postfix _channels32_days3 -t True --gpus 0,



#### f4 Scientific Name
# python main.py --name Scientific_name --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/lr1e-6/ --base configs/fishes/fishes_f4_clip.yaml --postfix _days2_lr1e-6 -t True --gpus 0,

# bioclip
# python main.py --name BioClip --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/ --base configs/fishes/fishes_f4_BioClip.yaml --postfix _days3 -t True --gpus 0,



#### f4 T2T
# python main.py --name T2T --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/lr1e-6/ --base configs/fishes/fishes_f4_t2t.yaml --postfix _days2_lr1e-6 -t True --gpus 0,

# 1280 encoding
# python main.py --name T2T --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/ --base configs/fishes/fishes_f4_t2t_1280.yaml --postfix _dim1280_days3 -t True --gpus 0,



#### f4 HLE
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/lr1e-6/ --base configs/fishes/fishes_f4_level_encoding.yaml --postfix _days2_lr1e-6 -t True --gpus 0,

###### Diff dimensions
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_256.yaml --postfix _dim256 -t True --gpus 0,

# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_128.yaml --postfix _dim128 -t True --gpus 0,

# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_64.yaml --postfix _dim64 -t True --gpus 0,

# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_768.yaml --postfix _dim768 -t True --gpus 0,

# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_32.yaml --postfix _dim32 -t True --gpus 0,

python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_1024.yaml --postfix _dim1024 -t True --gpus 0,

# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/diff_dimension/ --base configs/fishes/fishes_f4_hle_dim_16.yaml --postfix _dim16 -t True --gpus 0,

# #### f4 HLE leave one out
# python main.py --name HLE --logdir  /globalscratch/mridul/ldm/final_runs_eccv/fishes/ --base configs/fishes/fishes_f4_label_encoding_leave_out.yaml --postfix _days2_leaveout -t True --gpus 0,


################## multilevel
## 2 levels
# python main.py --name HLE --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level --base configs/fishes/fishes_f4_levels_2.yaml --postfix _days2_dim512_levels_2 -t True --gpus 0,

## 6 levels
# python main.py --name HLE --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level --base configs/fishes/fishes_f4_levels_6.yaml --postfix _days2_dim768_levels_6 -t True --gpus 0,

## 8 levels
# python main.py --name HLE --logdir /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level --base configs/fishes/fishes_f4_levels_8.yaml --postfix _days2_dim1024_levels_8 -t True --gpus 0,



exit;
