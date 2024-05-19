import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


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

# Placeholder for 38 classes embeddings
# Normally, embeddings should be extracted from a model or a similar source
# Here, we are creating random embeddings for demonstration purposes
num_classes = 38
embedding_dim = 100  # Example dimension of embeddings
embeddings = np.random.rand(num_classes, embedding_dim)

model_name = 'level_encoding'
ckpt_path = '/globalscratch/mridul/ldm/level_encoding/2023-12-03T09-33-45_HLE_f4_level_encoding_371/checkpoints/epoch=000119.ckpt'
config_path = '/globalscratch/mridul/ldm/level_encoding/2023-12-03T09-33-45_HLE_f4_level_encoding_371/configs/2023-12-03T09-33-45-project.yaml'


# # # #### CLIP f4
# model_name = 'scientific_name'
# config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/configs/2023-11-09T15-34-23-project.yaml'
# ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/checkpoints/epoch=000158.ckpt'

# # # #### BERT f4 node weighted len 256 | dim 512
# model_name = 'tree_to_text'
# config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-12-01T01-49-15_BERT_f4_max256_dim512/configs/2023-12-01T01-49-15-project.yaml'
# ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-12-01T01-49-15_BERT_f4_max256_dim512/checkpoints/epoch=000158.ckpt'

# ## class conditional
# model_name = 'class_conditional'
# config_path = "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/configs/2023-09-30T15-19-56-project.yaml" 
# ckpt_path = "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/checkpoints/epoch=000149.ckpt"


label_to_class_mapping = {0: 'Alosa-chrysochloris', 1: 'Carassius-auratus', 2: 'Cyprinus-carpio', 3: 'Esox-americanus', 
4: 'Gambusia-affinis', 5: 'Lepisosteus-osseus', 6: 'Lepisosteus-platostomus', 7: 'Lepomis-auritus', 8: 'Lepomis-cyanellus', 
9: 'Lepomis-gibbosus', 10: 'Lepomis-gulosus', 11: 'Lepomis-humilis', 12: 'Lepomis-macrochirus', 13: 'Lepomis-megalotis', 
14: 'Lepomis-microlophus', 15: 'Morone-chrysops', 16: 'Morone-mississippiensis', 17: 'Notropis-atherinoides', 
18: 'Notropis-blennius', 19: 'Notropis-boops', 20: 'Notropis-buccatus', 21: 'Notropis-buchanani', 22: 'Notropis-dorsalis', 
23: 'Notropis-hudsonius', 24: 'Notropis-leuciodus', 25: 'Notropis-nubilus', 26: 'Notropis-percobromus', 
27: 'Notropis-stramineus', 28: 'Notropis-telescopus', 29: 'Notropis-texanus', 30: 'Notropis-volucellus', 
31: 'Notropis-wickliffi', 32: 'Noturus-exilis', 33: 'Noturus-flavus', 34: 'Noturus-gyrinus', 35: 'Noturus-miurus', 
36: 'Noturus-nocturnus', 37: 'Phenacobius-mirabilis'}

def get_label_from_class(class_name):
    for key, value in label_to_class_mapping.items():
        if value == class_name:
            return key

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

config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, ckpt_path)  # TODO: check path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


if model_name in ['level_encoding', 'scientific_name', 'class_conditional']:
    class_to_node = '/fastscratch/mridul/fishes/class_to_ancestral_label.pkl'

elif model_name in ['tree_to_text']:
    class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs_weighted.pkl'

with open(class_to_node, 'rb') as pickle_file:
    class_to_node_dict = pickle.load(pickle_file)

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

embeddings_np = np.array([tensor.cpu().detach().numpy().flatten() for tensor in embeddings])

# breakpoint()


# Apply t-SNE transformation
# tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne = TSNE(n_components=2, learning_rate=50, verbose=1)
tsne_results = tsne.fit_transform(embeddings_np)

# # TNSE
# tsne = TSNE(n_components=2, learning_rate=150, verbose=2).fit_transform(X)
# tx, ty = tsne[:,0], tsne[:,1]
# tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
# ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

# Plotting
plt.figure(figsize=(16, 10))
for i in range(len(embeddings_np)):
    # breakpoint()
    plt.scatter(tsne_results[i, 0], tsne_results[i, 1])
    plt.text(tsne_results[i, 0], tsne_results[i, 1], label_to_class_mapping[i], fontdict={'weight': 'bold', 'size': 9})

plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.title('t-SNE Visualization of 38 Classes')

# Save the plot as an image file
output_path = f'/home/mridul/sample_ldm/tsne/{model_name}_try_fn.png'  # Specify your desired file path here
plt.savefig(output_path)

# Return the path where the image is saved
print(output_path)

# # TSNE computation
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(embeddings)

# # Plotting the TSNE results
# plt.figure(figsize=(16, 10))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
# plt.title('t-SNE plot for 38 classes')
# plt.xlabel('t-SNE axis 1')
# plt.ylabel('t-SNE axis 2')
# for i in range(num_classes):
#     plt.annotate(str(i), (tsne_results[i, 0], tsne_results[i, 1]))
# plt.show