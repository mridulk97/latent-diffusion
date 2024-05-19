# License: BSD
# Author: Sasank Chilamkurthy


import os
# from ldm.analysis_utils import get_phylomapper_from_config
# from ldm.data.phylogeny import Phylogeny
from ldm.plotting_utils import dump_to_json, plot_confusionmatrix, plot_confusionmatrix_colormap
import torch
import torch.nn as nn
from torchvision import datasets
import numpy
import albumentations

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch.nn.functional as F

from torchmetrics import F1Score
import matplotlib.pyplot as plt

import numpy as np


def get_input(x):
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    return x.float()

class Processor:
    def __init__(self, size):
        self.size = size
        
    def get_preprocess_image(self):
        def preprocess_image(image):
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = numpy.array(image).astype(numpy.uint8)
            
            rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            preprocessor = albumentations.Compose([rescaler, cropper])

            image = preprocessor(image=image)["image"]
            image = (image/127.5 - 1.0).astype(numpy.float32)
            return image
        
        return preprocess_image
        
@torch.no_grad()
def main(configs_yaml):
    DEVICE= configs_yaml.DEVICE
    size= configs_yaml.size
    bb_model_path = configs_yaml.bb_model_path
    dataset_path = configs_yaml.dataset_path
    save_path = configs_yaml.save_path
    batch_size= configs_yaml.batch_size
    num_workers= configs_yaml.num_workers
    phylogeny_path = configs_yaml.phylogeny_path
    file_name = configs_yaml.file_name
    mean_file_name = configs_yaml.mean_file_name
    std_file_name = configs_yaml.std_file_name
    
    
    dataset_test = datasets.ImageFolder(dataset_path, transform= Processor(size).get_preprocess_image())
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    
    indx_to_class = {v: k for k, v in dataset_test.class_to_idx.items()}

    num_classes = len(indx_to_class.keys())
    
    print("Loading model ...")
    model_ft = torch.load(bb_model_path)
    model_ft.eval()
    print("Model loaded ...")
    

    F1 = F1Score(num_classes=num_classes, multiclass=True, average='macro').to(DEVICE)

    
    preds = []
    classes = []
    for img, target in tqdm(dataloader):

        classes.append(target)
        preds.append(model_ft(get_input(img).to(DEVICE)))
        # breakpoint()
    classes = torch.cat(classes, dim=0).to(DEVICE)
    
    preds = torch.cat(preds, dim=0)
    # breakpoint()

    ## Softmax
    probabilities = F.softmax(preds, dim=1)
    probabilities_np = probabilities.detach().cpu().numpy()

    classes_numpy = classes.detach().cpu().numpy()

    def find_indices(lst, s):
        index_list =  [index for index, value in enumerate(lst) if value == s]
        return np.array(index_list)

    mean_list = []
    std_list = []
    # breakpoint()
    print(probabilities_np.shape)
    print(num_classes)
    np.save('/home/mridul/latent-diffusion/analysis/logits/trait_substitution_run3.npy', probabilities_np)
    breakpoint()
    # for i in range(num_classes):
    #     class_idx = find_indices(classes_numpy, i)
    #     class_sublist = classes_numpy[class_idx]
    #     preds_sublist = probabilities_np[class_idx]
    #     mean = np.mean(preds_sublist, axis=0)
    #     std = np.std(preds_sublist, axis=0)
    #     mean_list.append(mean)
    #     std_list.append(std)
    
    # final_mean = np.stack(mean_list, axis=0)
    # final_std = np.stack(std_list, axis=0)

    # print("Shape mean: ", final_mean.shape)
    # print("Shape std: ", final_std.shape)

    # np.save(mean_file_name, final_mean)
    # np.save(std_file_name, final_std)

    # print("F1 Score: ", F1(preds, classes).item())
    

    # # Plotting
    # num_classes = probabilities_np.shape[1]  # Assuming the second dimension represents class probabilities
    # fig, axes = plt.subplots(1, num_classes, figsize=(num_classes*5, 4))

    # for i in range(num_classes):
    #     ax = axes[i]
    #     # Extract probabilities for class i
    #     class_probs = probabilities_np[:, i]
        
    #     # Plot as a histogram
    #     ax.hist(class_probs, bins=1, alpha=0.7, label=f'Class {i}')
    #     ax.set_title(f'Class {i} Probability Distribution')
    #     ax.set_xlabel('Probability')
    #     ax.set_ylabel('Frequency')
    #     ax.legend()

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/test_resnet-logits.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    print(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)