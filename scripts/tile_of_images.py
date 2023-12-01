from PIL import Image
import os

def create_image_tile_without_labels(base_directory, output_filename):
    """
    Create a tile of images from different subdirectories, without any labels.

    Each subdirectory in the base_directory represents a model,
    and contains 5 images with the same names across all models.

    The resulting image tile will have one column per model and
    one row per species (image), with each cell showing the image
    of that species for the corresponding model.

    Args:
    - base_directory (str): The path to the directory containing model subdirectories.
    - output_filename (str): The filename to save the tiled image.
    """

    # List all model subdirectories
    # model_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    model_dirs = ['gt', 'class_conditional', 'scientific_name', 'tree_to_text', 'level_encoding']

    # Assume all subdirectories have the same set of image filenames
    image_filenames = os.listdir(os.path.join(base_directory, model_dirs[0]))

    # Initialize list to hold all rows
    rows = []
    crop_height = 0  # Height to crop from the top and bottom of each image

    # Process each species image
    for image_name in image_filenames:
        row_images = []

        for model_dir in model_dirs:
            image_path = os.path.join(base_directory, model_dir, image_name)
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
final_image = '/home/mridul/sample_ldm/handpicked/handpicked_tile_label_remove_pixels_0.png'
# Example usage
create_image_tile_without_labels(handpicked_images, final_image)
