import argparse, os, sys, glob
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from fvcore.nn import FlopCountAnalysis, flop_count_table


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    # pl_sd = torch.load(ckpt, map_location="cpu")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

ckpt_path = '/globalscratch/mridul/ldm/final_runs_eccv/fishes/2024-03-01T23-15-36_HLE_days3/checkpoints/epoch=000119.ckpt'
config_path = '/globalscratch/mridul/ldm/final_runs_eccv/fishes/2024-03-01T23-15-36_HLE_days3/configs/2024-03-01T23-15-36-project.yaml'

config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, ckpt_path)  # TODO: check path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

noise = torch.randn(8, 3, 256, 256)
noise = noise.to(device)
cc = torch.load('/home/mridul/latent-diffusion/analysis/c_tensor.pt', map_location=torch.device('cpu'))

flops = FlopCountAnalysis(model, (noise, cc))
breakpoint()
print(flops.total())


breakpoint()
print('here')