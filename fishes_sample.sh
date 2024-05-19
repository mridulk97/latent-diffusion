#!/bin/bash
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=0-02:00:00 # 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mridul/latent-diffusion/slurm/multilevel/sample/slurm-%j-%x.out


module reset

module load Anaconda3/2020.11

source activate ldm
module reset
which python

python scripts/fish_sample.py --outdir /globalscratch/mridul/ldm/final_runs_eccv/fishes_samples/multi_level/ 
--output_dir_name levels_4 --n_samples 100 --ckpt_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/2024-03-01T23-15-36_HLE_days3/checkpoints/epoc
h=000119.ckpt  --config_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/2024-03-01T23-15-36_HLE_days3/configs/2024-03-01T23-15-36-project.yaml --node_dict /fastscratch/mridul/fishes/class_to_ancestral_label.pkl

################## multilevel
## 2 levels
# python scripts/fish_sample.py --outdir /globalscratch/mridul/ldm/final_runs_eccv/fishes_samples/multi_level/ --output_dir_name test_level2_1 --n_samples 100 --ckpt_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-46-53_HLE_days2_dim512_levels_2/checkpoints/epoch=000119.ckpt --config_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-46-53_HLE_days2_dim512_levels_2/configs/2024-05-12T20-46-53-project.yaml --node_dict /projects/ml4science/mridul/data/fishes_all/fish_levels/fishes_levels_2.pkl

## 6 levels
python scripts/fish_sample.py --outdir /globalscratch/mridul/ldm/final_runs_eccv/fishes_samples/multi_level/ --output_dir_name levels_6 --n_samples 100 --ckpt_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-47-24_HLE_days2_dim768_levels_6/checkpoints/epoch=000119.ckpt --config_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-47-24_HLE_days2_dim768_levels_6/configs/2024-05-12T20-47-24-project.yaml --node_dict /projects/ml4science/mridul/data/fishes_all/fish_levels/fishes_levels_6.pkl

## 8 levels 512 dim
# python scripts/fish_sample.py --outdir /globalscratch/mridul/ldm/final_runs_eccv/fishes_samples/multi_level/ --output_dir_name levels_8 --n_samples 100 --ckpt_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-47-24_HLE_days2_dim512_levels_8/checkpoints/epoch=000119.ckpt --config_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-47-24_HLE_days2_dim512_levels_8/configs/2024-05-12T20-47-24-project.yaml --node_dict /projects/ml4science/mridul/data/fishes_all/fish_levels/fishes_levels_8.pkl

## 8 levels 1024 dim
# python scripts/fish_sample.py --outdir /globalscratch/mridul/ldm/final_runs_eccv/fishes_samples/multi_level/ --output_dir_name levels_8 --n_samples 100 --ckpt_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-48-06_HLE_days2_dim1024_levels_8/checkpoints/epoch=000119.ckpt --config_path /globalscratch/mridul/ldm/final_runs_eccv/fishes/multi_level/2024-05-12T20-48-06_HLE_days2_dim1024_levels_8/configs/2024-05-12T20-48-06-project.yaml --node_dict /projects/ml4science/mridul/data/fishes_all/fish_levels/fishes_levels_8.pkl



exit;
