import os
import cv2
import numpy as np

label_to_class_mapping = {0: 'Alosa-chrysochloris', 1: 'Carassius-auratus', 2: 'Cyprinus-carpio', 3: 'Esox-americanus', 
4: 'Gambusia-affinis', 5: 'Lepisosteus-osseus', 6: 'Lepisosteus-platostomus', 7: 'Lepomis-auritus', 8: 'Lepomis-cyanellus', 
9: 'Lepomis-gibbosus', 10: 'Lepomis-gulosus', 11: 'Lepomis-humilis', 12: 'Lepomis-macrochirus', 13: 'Lepomis-megalotis', 
14: 'Lepomis-microlophus', 15: 'Morone-chrysops', 16: 'Morone-mississippiensis', 17: 'Notropis-atherinoides', 
18: 'Notropis-blennius', 19: 'Notropis-boops', 20: 'Notropis-buccatus', 21: 'Notropis-buchanani', 22: 'Notropis-dorsalis', 
23: 'Notropis-hudsonius', 24: 'Notropis-leuciodus', 25: 'Notropis-nubilus', 26: 'Notropis-percobromus', 
27: 'Notropis-stramineus', 28: 'Notropis-telescopus', 29: 'Notropis-texanus', 30: 'Notropis-volucellus', 
31: 'Notropis-wickliffi', 32: 'Noturus-exilis', 33: 'Noturus-flavus', 34: 'Noturus-gyrinus', 35: 'Noturus-miurus', 
36: 'Noturus-nocturnus', 37: 'Phenacobius-mirabilis'}

def find_key_by_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None

def load_images_from_directory(root_dir):
    image_data = []
    label_data = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                specie = file.split('_')[0]
                label = find_key_by_value(label_to_class_mapping, specie)
                img_path = os.path.join(subdir, file)
                image = cv2.imread(img_path)
                if image is not None:
                    image_data.append(image)
                    label_data.append(label)

    return np.array(image_data), np.array(label_data)


if __name__ == "__main__":
    main_dir = "/fastscratch/mridul/fishes/fishes_test_diffusion_new"
    output_file = "/fastscratch/mridul/fishes/fishes_test_diffusion_new.npz"
    
    images, labels = load_images_from_directory(main_dir)
    np.savez(output_file, images, labels)
    print(f"{len(images)} images saved to {output_file}")


