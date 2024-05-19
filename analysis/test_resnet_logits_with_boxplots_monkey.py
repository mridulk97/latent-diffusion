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
import pickle


def group_continuous_spans(values):
    sorted_values = sorted(values)  # Ensure the values are sorted
    grouped_spans = []
    current_span = []

    for value in sorted_values:
        # If current_span is empty or value is consecutive, append to current_span
        if not current_span or value == current_span[-1] + 1:
            current_span.append(value)
        else:
            # Once a gap is found, save the first and last of the current_span and start a new one
            if len(current_span) > 1:
                grouped_spans.append([current_span[0], current_span[-1]])
            else:
                grouped_spans.append(current_span)
            current_span = [value]

    # Add the last span to the list if it's not empty
    if current_span:
        if len(current_span) > 1:
            grouped_spans.append([current_span[0], current_span[-1]])
        else:
            grouped_spans.append(current_span)

    return grouped_spans

# ancestor_level3 = { 0: ['Alosa chrysochloris'],
#                     1: ['Carassius auratus', 'Cyprinus carpio'],
#                     2: ['Esox americanus'],
#                     3: ['Gambusia affinis'], 
#                     4: ['Lepisosteus osseus', 'Lepisosteus platostomus'],
#                     5: ['Lepomis auritus','Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
#                         'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis','Lepomis microlophus'],
#                     6: ['Morone chrysops', 'Morone mississippiensis'],

#                     7: ['Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 
#                         'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 
#                         'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 
#                         'Notropis stramineus','Notropis telescopus', 'Notropis texanus', 
#                         'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
                    
#                     8: ['Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
#                         'Noturus nocturnus']
#                 }

# ancestor_level1 = { 0: ['Alosa chrysochloris', 'Carassius auratus', 'Cyprinus carpio',
#                         'Notropis atherinoides', 'Notropis blennius', 'Notropis boops',
#                         'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 
#                         'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 
#                         'Notropis percobromus', 'Notropis stramineus','Notropis telescopus', 
#                         'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 
#                         'Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
#                         'Noturus nocturnus','Phenacobius mirabilis'],
#                     1: ['Esox americanus', 'Gambusia affinis', 'Lepomis auritus',
#                         'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
#                         'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 
#                         'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
#                     2: ['Lepisosteus osseus', 'Lepisosteus platostomus']
#                 }

# ancestor_level2 = { 0: ['Alosa chrysochloris'],
#                     1: ['Carassius auratus', 'Cyprinus carpio', 'Notropis atherinoides', 
#                         'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 
#                         'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 
#                         'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 
#                         'Notropis stramineus','Notropis telescopus', 'Notropis texanus', 
#                         'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
#                     2: ['Esox americanus'],
#                     3: ['Gambusia affinis', 'Lepomis auritus',
#                         'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
#                         'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 
#                         'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
#                     4: ['Lepisosteus osseus', 'Lepisosteus platostomus'],
#                     5: ['Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
#                         'Noturus nocturnus']
#                 }

ancestor_level1 = {0: ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 
                       't10', 't11', 't12', 't13', 't14', 't15', 't16', 't9'],
                    1: ['t17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 
                        't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32']}

ancestor_level2 = {0: ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'],
  1: ['t10', 't11', 't12', 't13', 't14', 't15', 't16', 't9'],
  2: ['t17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'],
  3: ['t25', 't26', 't27', 't28', 't29', 't30', 't31', 't32']}

ancestor_level3 =  {0: ['t1', 't2', 't3', 't4'],
  1: ['t10', 't11', 't12', 't9'],
  2: ['t13', 't14', 't15', 't16'],
  3: ['t17', 't18', 't19', 't20'],
  4: ['t21', 't22', 't23', 't24'],
  5: ['t25', 't26', 't27', 't28'],
  6: ['t29', 't30', 't31', 't32'],
  7: ['t5', 't6', 't7', 't8']}

all_species = ['Alosa chrysochloris', 'Carassius auratus', 'Cyprinus carpio', 'Esox americanus',
               'Gambusia affinis', 'Lepisosteus osseus', 'Lepisosteus platostomus', 
               'Lepomis auritus','Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
               'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis','Lepomis microlophus',
               'Morone chrysops', 'Morone mississippiensis',
               'Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 
                        'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 
                        'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 
                        'Notropis stramineus','Notropis telescopus', 'Notropis texanus', 
                        'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis',
                        'Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
                        'Noturus nocturnus']

monkeys_mapping = '/projects/ml4science/mridul/data/monkeys_dataset/monkey_HLE_labels.pkl'

with open(monkeys_mapping, 'rb') as f:
    monkeys_hle = pickle.load(f)

all_monkyes = sorted([key for key in monkeys_hle])

class_to_idx = {}
for idx, species in enumerate(all_monkyes):
    class_to_idx[species] = idx

speceis_level = {value: [key] for key, value in class_to_idx.items()}


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
    box_plot_dir = configs_yaml.box_plot_dir
    
    
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
    classes = torch.cat(classes, dim=0).to(DEVICE)
    
    preds = torch.cat(preds, dim=0)

    ## Softmax
    probabilities = F.softmax(preds, dim=1)
    probabilities_np = probabilities.detach().cpu().numpy()

    classes_numpy = classes.detach().cpu().numpy()

    def find_indices(lst, s):
        index_list =  [index for index, value in enumerate(lst) if value == s]
        return np.array(index_list)

    mean_list = []
    std_list = []
    # np.save('/home/mridul/latent-diffusion/analysis/logits/all_preds.npy', probabilities_np)

    def create_box_plot(data, destination_path, ancestry_mapping, i):
        ig, ax = plt.subplots(figsize=(10, 12))
        # boxplots = ax.boxplot(data, labels=all_species, patch_artist=True, showfliers=False)
        boxplots = ax.boxplot(data, patch_artist=True, showfliers=False, vert=False, widths=0.8)
        class_names = ancestry_mapping[i]



        ax.set_yticklabels(all_monkyes, rotation=0, ha="right")

        values_for_keys = [class_to_idx[key] for key in class_names if key in class_to_idx]


        ax.set_xlabel('Probabilities', fontsize=20)
        ax.set_xlim([-0.01,1])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=24)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(boxplots[element], color='black')

        for patch in boxplots['boxes']:
            patch.set(facecolor='pink')  
        # ax.set_title(f'Distribution for Classes')
        
        # box_width = 0.8 / len(data)  # Assuming an even distribution of boxes
        # for index in values_for_keys:
        #     left_edge = index + 1 - box_width / 2  # Adjust to center the highlight on the box
        #     right_edge = index + 1 + box_width / 2
        #     ax.axvspan(left_edge, right_edge, color='red', alpha=0.3)


        cont_spans = group_continuous_spans(values_for_keys)
        # print(i, values_for_keys)
        for span in cont_spans:
            # print(span)
            if len(span) == 1:
                left_edge = span[0] + 0.5  # Adjust as needed for precise positioning
                right_edge = span[0] + 1.5
            else:
                left_edge = span[0] + 0.5  # Adjust as needed for precise positioning
                right_edge = span[1] + 1.5

            ax.axhspan(left_edge, right_edge, color='green', alpha=0.2)



        ax.xaxis.grid(True)
        plt.tight_layout()
        # Save each plot with a unique name
        final_path = os.path.join(destination_path)
        os.makedirs(final_path, exist_ok=True)
        plt.savefig(f'{final_path}/class_{i}')


    for i in range(num_classes):
        class_idx = find_indices(classes_numpy, i)
        class_sublist = classes_numpy[class_idx]
        preds_sublist = probabilities_np[class_idx]
        create_box_plot(preds_sublist, box_plot_dir, ancestor_level1, i)
        # create_box_plot(preds_sublist, box_plot_dir, speceis_level, i)
    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/test_resnet-logits-boxplot_monkey.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    print(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)