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

# class_names = ['level1_cyprinus', 'level1_morone', 'level1_notropis',
#                 'level2_gambusia_to_esox', 'level2_morone_to_esox', 'level2_notuturs_to_notropis',
#                 'level3_gambusia_to_morone', 'level3_lepomis_to_morone', 'level3_morone_to_gambusia']

class_names = ['level3_carassius_to_notropis', 'level3_morone_to_lepomis', 'level3_lepomis_to_morone', 
               'level3_notropis_to_carassius', 'level2_notuturs_to_notropis', 'level2_gambusia_to_esox', 
               'level2_notropis_to_notuturs', 'level2_lepomis_to_esox', 'level2_esox_to_gambusia']

class_names = ['level3_carassius_to_notropis', 'level3_morone_to_lepomis', 'level3_lepomis_to_morone', 
               'level3_notropis_to_carassius', 'level2_notuturs_to_notropis', 'level2_gambusia_to_esox', 
               'level2_notropis_to_notuturs', 'level2_lepomis_to_esox', 'level2_esox_to_gambusia']

class_names = sorted(class_names)

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

def plot_boxplot(data, destination_path, class_specific):

        ig, ax = plt.subplots(figsize=(10, 12))
        boxplots = ax.boxplot(data, patch_artist=True, showfliers=False, vert=False, widths=0.8)
        # breakpoint()
        # class_names = ancestry_mapping[i]


        ax.set_yticklabels(all_species, rotation=0, ha="right")

        values_for_keys = [class_to_idx[key] for key in class_names if key in class_to_idx]



        ax.set_xlabel('Probabilities', fontsize=18)
        ax.set_xlim([-0.01,1])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=24)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(boxplots[element], color='black')

        for patch in boxplots['boxes']:
            patch.set(facecolor='pink')  

        

        cont_spans = group_continuous_spans(values_for_keys)

        if class_specific =='level2_gambusia_to_esox':
            cont_spans = [[4]]
        elif class_specific =='level2_notuturs_to_notropis':
            cont_spans = [[32]]
        elif class_specific =='level3_lepomis_to_morone':
            cont_spans = [[10]]
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
        plt.savefig(f'{final_path}/{class_specific}')


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
# mean_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_mean.npy')
# std_ref = np.load('/home/mridul/latent-diffusion/analysis/logits/ref_level4_std.npy')
# all_classes = range(38)
# plot_barplot(all_classes, mean_ref, std_ref, '/home/mridul/latent-diffusion/analysis/images/barplot/ref_level4', speceis_level)

# substituions = np.load('/home/mridul/latent-diffusion/analysis/logits/trait_substitution.npy')
substituions = np.load('/home/mridul/latent-diffusion/analysis/logits/trait_substitution_run3.npy')
destination_path = '/home/mridul/latent-diffusion/analysis/images/boxplot/substitutions/run3'

for i in range(9):
    # breakpoint()
    class_specific = class_names[i]
    left = i*100
    right = left + 100
    data = substituions[left:right]
    plot_boxplot(data, destination_path, class_specific)
    print(i, left, right, data.shape)
     
