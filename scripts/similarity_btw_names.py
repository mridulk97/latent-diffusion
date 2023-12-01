import argparse, os, sys, glob
import torch
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt

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

 

# #### CLIP f8
# config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/configs/2023-11-09T15-30-05-project.yaml'
# ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/checkpoints/epoch=000119.ckpt'

# #### CLIP f4
# model_name = 'scientific_name'
# config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/configs/2023-11-09T15-34-23-project.yaml'
# ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/checkpoints/epoch=000158.ckpt'


# #### BERT f4 node weighted 256
model_name = 'tree_to_text'
config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/configs/2023-11-13T23-08-55-project.yaml'
ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/checkpoints/epoch=000119.ckpt'

#### Level Encoding
# model_name = 'level_encoding'
# config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_TEST_f4_ancestral_label_encoding/configs/2023-11-13T23-08-55-project.yaml'
# ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_TEST_f4_ancestral_label_encoding/checkpoints/epoch=000119.ckpt'

### class conditional
# model_name = 'class_conditional'
# config_path = "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/configs/2023-09-30T15-19-56-project.yaml" 
# ckpt_path = "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/checkpoints/epoch=000149.ckpt"


# class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs.pkl'

if model_name in ['level_encoding', 'scientific_name', 'class_conditional']:
    class_to_node = '/fastscratch/mridul/fishes/class_to_ancestral_label.pkl'

elif model_name in ['tree_to_text']:
    class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs_weighted.pkl'

with open(class_to_node, 'rb') as pickle_file:
    class_to_node_dict = pickle.load(pickle_file)


config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, ckpt_path)  # TODO: check path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

n_samples =1
embeddings = []
sorted_class_to_node_dict = dict(sorted(class_to_node_dict.items()))
classes = sorted(range(38))

for class_name, node_representation in tqdm(sorted_class_to_node_dict.items()):
# for class_label in tqdm(classes):
    # prompt = node_representation
    # promt = class_name
    if model_name in ['level_encoding', 'tree_to_text']:
        prompt = node_representation
    elif model_name in ['class_conditional']:
        # prompt = class_label

        prompt = [int(num) for num in node_representation[0].split(', ')][-1]
    elif model_name == 'scientific_name':
        prompt = class_name
    # prompt = class_label
    print(prompt)
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            # uc = model.get_learned_conditioning(3 * [""])
            for n in trange(1, desc="Sampling"):
                all_prompts = n_samples * [prompt]

                if model_name in ['class_conditional']:
                    xc = torch.tensor(n_samples*[prompt])
                    c = model.get_learned_conditioning({'class': xc.to(model.device)})

                elif model_name in ['level_encoding']:
                    xc = n_samples * (prompt)
                    xc = [tuple(xc)]
                    c = model.get_learned_conditioning({'class_to_node': xc})

                else:
                    c = model.get_learned_conditioning(n_samples * [prompt])

                embeddings.append(c[0])

                print(c[0].shape)

n=38
distance_matrix = torch.zeros((n, n))
for i in range(n):
    for j in range(n):

        cosine_dist = 1 - cosine_similarity(embeddings[i], embeddings[j])

        if model_name in ['tree_to_text']:
            ## for mean across 256
            distance_matrix[i, j] = torch.reshape(cosine_dist.mean(), (1,))
            ## for cls token
            # similarity_matrix[i, j] = cosine_dist[0]
        else:
            distance_matrix[i, j] = cosine_dist

        # similarity_matrix[i, j] = 1 - cosine_similarity(clip_embeddings[i], clip_embeddings[j], dim=1)

# Plotting the grid

normalized_array = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

plt.figure(figsize=(10, 8))
heat_map = sns.heatmap(normalized_array, cmap='coolwarm', annot=False)

cbar = heat_map.collections[0].colorbar
cbar.ax.tick_params(labelsize=40)

# Calculate the tick positions and labels for every 5th column
xtick_positions = np.arange(0, normalized_array.shape[1], 5)
xtick_labels = [str(x) for x in xtick_positions]

# Calculate the tick positions and labels for every 5th row
ytick_positions = np.arange(0, normalized_array.shape[0], 5)
ytick_labels = [str(y) for y in ytick_positions]

# Set x-ticks and y-ticks
plt.xticks(ticks=xtick_positions, labels=xtick_labels, fontsize=25)
plt.yticks(ticks=ytick_positions, labels=ytick_labels, fontsize=25)

# plt.title("Cosine Distance Grid")
# plt.xlabel("Tensor Index")
# plt.ylabel("Tensor Index")





os.makedirs('/home/mridul/sample_ldm/similarity_norm', exist_ok=True)
file_path = '/home/mridul/sample_ldm/similarity_norm/{}.png'.format(model_name)

plt.tight_layout()
plt.savefig(file_path)
plt.show()

