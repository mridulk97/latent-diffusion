import os
import numpy as np
import matplotlib.pyplot as plt


ancestor_level3 = { 0: ['Alosa chrysochloris'],
                    1: ['Carassius auratus', 'Cyprinus carpio'],
                    2: ['Esox americanus'],
                    3: ['Gambusia affinis'], 
                    4: ['Lepisosteus osseus', 'Lepisosteus platostomus'],
                    5: ['Lepomis auritus','Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
                        'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis','Lepomis microlophus'],
                    6: ['Morone chrysops', 'Morone mississippiensis'],

                    7: ['Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 
                        'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 
                        'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 
                        'Notropis stramineus','Notropis telescopus', 'Notropis texanus', 
                        'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
                    
                    8: ['Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
                        'Noturus nocturnus']
                }

ancestor_level1 = { 0: ['Alosa chrysochloris', 'Carassius auratus', 'Cyprinus carpio',
                        'Notropis atherinoides', 'Notropis blennius', 'Notropis boops',
                        'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 
                        'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 
                        'Notropis percobromus', 'Notropis stramineus','Notropis telescopus', 
                        'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 
                        'Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
                        'Noturus nocturnus','Phenacobius mirabilis'],
                    1: ['Esox americanus', 'Gambusia affinis', 'Lepomis auritus',
                        'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
                        'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 
                        'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
                    2: ['Lepisosteus osseus', 'Lepisosteus platostomus']
                }

ancestor_level2 = { 0: ['Alosa chrysochloris'],
                    1: ['Carassius auratus', 'Cyprinus carpio', 'Notropis atherinoides', 
                        'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 
                        'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 
                        'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 
                        'Notropis stramineus','Notropis telescopus', 'Notropis texanus', 
                        'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
                    2: ['Esox americanus'],
                    3: ['Gambusia affinis', 'Lepomis auritus',
                        'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus',
                        'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 
                        'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
                    4: ['Lepisosteus osseus', 'Lepisosteus platostomus'],
                    5: ['Noturus exilis', 'Noturus flavus','Noturus gyrinus', 'Noturus miurus', 
                        'Noturus nocturnus']
                }

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

all_species = sorted(all_species)

class_to_idx = {}
for idx, species in enumerate(all_species):
    class_to_idx[species] = idx

def get_values_from_keys(ancestor_dict, values):
    pass


def plot_barplot(classes, mean, std, destination_path, ancestry_mapping):

    for i in range(mean.shape[0]):
        values_for_keys = [class_to_idx[key] for key in ancestry_mapping[i] if key in class_to_idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(classes, mean[i], yerr=std[i], align='center', alpha=0.5, ecolor='darkgray', capsize=2)
        ax.set_ylabel('Mean/Std')
        ax.set_xticks(classes)
        ax.set_title(f'Class {i}')
        ax.yaxis.grid(True)

        # Highlight specific bars by changing their properties
        for index in values_for_keys:
            bars[index].set_alpha(1.0)  # Increase alpha to make it more solid
            bars[index].set_color('r') 

        # Save each plot with a unique name
        final_path = os.path.join(destination_path)
        os.makedirs(final_path, exist_ok=True)
        plt.savefig(f'{final_path}/class_{i}')


speceis_level = {value: [key] for key, value in class_to_idx.items()}
mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_mean.npy')
std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_std.npy')
all_classes = range(38)
plot_barplot(all_classes, mean_ref, std_ref, '/home/mridul/latent-diffusion/analysis/images/barplot/ref_level4', speceis_level)


# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level3_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level3_std.npy')
# all_classes = range(38)
# plot_barplot(all_classes, mean_ref, std_ref, '/home/mridul/latent-diffusion/analysis/images/barplot/level3', ancestor_level3)

# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level2_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level2_std.npy')
# all_classes = range(38)
# plot_barplot(all_classes, mean_ref, std_ref, '/home/mridul/latent-diffusion/analysis/images/barplot/level2', ancestor_level2)

# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level1_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level1_std.npy')
# all_classes = range(38)
# plot_barplot(all_classes, mean_ref, std_ref, '/home/mridul/latent-diffusion/analysis/images/barplot/level1', ancestor_level1)









# def plot_graphs(array_to_plot, classes, destination_path, mean_or_std, ancestry_mapping):

#     for i in range(array_to_plot.shape[0]):
#         values_for_keys = [class_to_idx[key] for key in ancestry_mapping[i] if key in class_to_idx]


#         plt.figure(figsize=(10, 6))
#         plt.plot(classes, array_to_plot[i], marker='o', linestyle='-', color='b')  # Plot the i-th instance's values
#         if mean_or_std == 'mean':
#             plt.title(f'Class {i} Mean')
#         elif mean_or_std == 'std':
#             plt.title(f'Class {i} Std')
#         plt.xlabel('Class Index')
#         plt.ylabel('Value')
#         plt.xticks(classes)
#         plt.grid(True)
        
#         # Save each plot with a unique name
#         final_path = os.path.join(destination_path, mean_or_std)
#         os.makedirs(final_path, exist_ok=True)
#         plt.savefig(f'{final_path}/class_{i}')

# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_std.npy')
# all_classes = range(mean_ref.shape[0])
# plot_graphs(mean_ref, all_classes, '/home/mridul/latent-diffusion/analysis/images/ref_level4', 'mean')
# plot_graphs(mean_ref, all_classes, '/home/mridul/latent-diffusion/analysis/images/ref_level4', 'std')
        

# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level3_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/level3_std.npy')
# all_classes = range(38)
# plot_graphs(mean_ref, all_classes, '/home/mridul/latent-diffusion/analysis/images/level3', 'mean', ancestor_level3)
# plot_graphs(mean_ref, all_classes, '/home/mridul/latent-diffusion/analysis/images/level3', 'std', ancestor_level3)