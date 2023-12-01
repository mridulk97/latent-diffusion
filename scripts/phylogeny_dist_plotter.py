import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# phylogeny_distance_level3 = '/home/mridul/phylonn/analysis/phylo analysis/level3.csv'
# df = pd.read_csv(phylogeny_distance_level3)
# df = df.drop(['Unnamed: 0'], axis=1)
# dist_matrix = df.to_numpy()


# # Plotting the grid

# plt.figure(figsize=(10, 8))
# # sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, vmin=0, vmax=2)
# sns.heatmap(dist_matrix, cmap='coolwarm', annot=False)
# plt.title("Cosine Distance Grid")
# plt.xlabel("Tensor Index")
# plt.ylabel("Tensor Index")
# plt.show()


# os.makedirs('/home/mridul/sample_ldm/similarity', exist_ok=True)
# # Saving the plot
# file_path = '/home/mridul/sample_ldm/similarity/phylogeny_dist.png'
# plt.savefig(file_path)

class_to_node = '/fastscratch/mridul/fishes/class_to_ancestral_label.pkl'
with open(class_to_node, 'rb') as pickle_file:
    class_to_node_dict = pickle.load(pickle_file)

all_classes = []
for class_name, node_representation in (class_to_node_dict.items()):
    all_classes.append(class_name)

all_classes = sorted(all_classes)


from ldm.data.phylogeny import Phylogeny
phylogeny = Phylogeny("/fastscratch/elhamod/data/Fish", all_classes)


tree = phylogeny.tree

dist = np.zeros(shape=(38, 38))

for i in range(38):
    for j in range(38):
        if i!=j:
            i_th = 'ott' + str(phylogeny.ott_id_dict[all_classes[i]])
            j_th = 'ott' + str(phylogeny.ott_id_dict[all_classes[j]])
            dist[i][j] = tree.get_distance(i_th, j_th)

normalized_array = (dist - dist.min()) / (dist.max() - dist.min())
# breakpoint()

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


os.makedirs('/home/mridul/sample_ldm/similarity_final_small', exist_ok=True)
# Saving the plot
file_path = '/home/mridul/sample_ldm/similarity_final_small/phylogeny_dist.png'

plt.tight_layout()
plt.savefig(file_path)
plt.show()


print('Yay')