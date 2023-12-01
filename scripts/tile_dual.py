from PIL import Image
import os

def create_image_tile_side_by_side_species(base_directory, output_filename):
    """
    Create a tile of images from different subdirectories, placing images for the same species 
    side by side for each model, resulting in a 5x10 grid assuming 5 different species.

    Each subdirectory in the base_directory represents a model,
    and each species is represented by two images named 'image_1.png' and 'image_2.png',
    where 'image' is the species name.

    The resulting image tile will have two columns per model (side by side for each species) 
    and one row per species, with each cell showing the image of that species for the 
    corresponding model.

    Args:
    - base_directory (str): The path to the directory containing model subdirectories.
    - output_filename (str): The filename to save the tiled image.
    """

    # List all model subdirectories
    # model_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    model_dirs = ['gt', 'class_conditional', 'scientific_name', 'tree_to_text', 'level_encoding']

    # Assume the first subdirectory has the complete set of species
    species_names = set(filename.split('_')[0] for filename in os.listdir(os.path.join(base_directory, model_dirs[0])))
    crop_height = 20  # Height to crop from the top and bottom of each image

    # Initialize list to hold all rows
    rows = []

    # Process each species
    for species in species_names:
        row_images = []

        for model_dir in model_dirs:
            for image_num in range(1, 3):  # For 'image_1.png' and 'image_2.png'
                image_path = os.path.join(base_directory, model_dir, f"{species}_{image_num}.png")
                image = Image.open(image_path)
                # Cropping the image
                image = image.crop((0, crop_height, image.width, image.height - crop_height))
                row_images.append(image)

        # Concatenate images in this row
        total_row_width = sum(img.width for img in row_images)
        row_image = Image.new('RGB', (total_row_width, row_images[0].height))
        x_offset = 0
        for img in row_images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width

        rows.append(row_image)

    # Concatenate all rows to create the final tile
    max_row_width = max(img.width for img in rows)
    total_tile_height = sum(img.height for img in rows)
    tile_image = Image.new('RGB', (max_row_width, total_tile_height))

    y_offset = 0
    for row in rows:
        tile_image.paste(row, (0, y_offset))
        y_offset += row.height

    # Save the final image
    tile_image.save(output_filename)


# model_dirs = ['gt', 'class_conditional', 'scientific_name', 'tree_to_text', 'level_encoding']

handpicked_images = '/home/mridul/sample_ldm/handpicked'
final_image = '/home/mridul/sample_ldm/handpicked/tile/2images.png'
# Example usage
create_image_tile_side_by_side_species(handpicked_images, final_image)
