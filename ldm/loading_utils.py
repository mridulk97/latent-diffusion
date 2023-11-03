#based on https://github.com/CompVis/taming-transformers

import yaml
# from ldm.models.autoencoder import VQModel
# from ldm.models.cwautoencoder import CWmodelVQGAN
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config

######### loaders

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    # breakpoint()
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config_path, ckpt_path=None):
# def load_model(config_path, ckpt_path=None, cuda=False, model_type=VQModel):
    # breakpoint()
    # model = model_type(**config.model.params)
    # if ckpt_path is not None:
    #     sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    #     missing, unexpected = model.load_state_dict(sd, strict=True)
    # if cuda:
    #     model = model.cuda()

    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, ckpt_path)
    breakpoint()
    return model