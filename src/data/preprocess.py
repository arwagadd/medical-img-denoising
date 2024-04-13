from typing import List, Generator, Optional
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str, batch_size: int) -> Generator[List[np.ndarray], None, None]:
    image_dir = os.path.join(path, "dbs", "raw", "archive", "GAN-Traning Images")

    if not os.path.isdir(image_dir):
        print(f"Directory {image_dir} does not exist")
        return

    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    all_images = []
    target_size = (300, 300)  # target size to resize images

    for file_name in image_files:
        file_path = os.path.join(image_dir, file_name)
        image = cv2.imread(file_path)

        if image is not None:
            # resize the image
            image = cv2.resize(image, target_size)

            all_images.append(image)
    start_idx = 0
    while start_idx < len(all_images):
        end_idx = min(start_idx + batch_size, len(all_images))
        batch_images = all_images[start_idx:end_idx]

        # Resize images to the maximum dimensions
        resized_images = []
        for image in batch_images:
            resized_image = cv2.resize(image, target_size)
            resized_images.append(resized_image)

        start_idx = end_idx
        yield resized_images

# original images
def convert_to_numpy(batch_images: List[np.ndarray], output_file: str):
    batch_array = np.stack(batch_images)  # Stack the list of images into a single array
    np.save(output_file, batch_array)


def unify_shape(X: list[np.ndarray]) -> np.ndarray:
    ...

def gen_y(X: List[np.ndarray]) -> List[tuple[np.ndarray, np.ndarray]]:
    noisy_images = []
    for x in X:
        noise = np.random.normal(loc=0, scale=29, size=x.shape[:2])
        noisy_image = x + np.expand_dims(noise, axis=-1)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        noisy_images.append(noisy_image)
    
    original_images = np.stack(X)
    return list(zip(original_images, noisy_images))



def save_data(X: np.ndarray, Y: np.ndarray, path: str, batch_index: int):
    ...
    # Save X and Y as x.npy and y.npy

    # save 2 files: x.npy y.npy
    # x.npy: dim[x]=[batch,w,h,c]  noised img
    # y.npy: dim[y]=[batch,w,h,c]  denoised/unified 


def main():
    raw_data_path = ""  # Update with the path to your raw data
    batch_size = 9

    # Load the original images and resize them
    loaded_data = load_data(raw_data_path, batch_size)
    output_dir = "./dbs/intermittent"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reset the generator to load the original images again
    loaded_data = load_data(raw_data_path, batch_size)

    # Save resized original images as numpy files (x_batchIndex.npy)
    for i, batch_images in enumerate(loaded_data):
        convert_to_numpy(batch_images, os.path.join(output_dir, f"x_batch_{i}.npy"))

    # Reset the generator to load the original images again
    loaded_data = load_data(raw_data_path, batch_size)

    # Generate noisy images and save them in the same way as x
    for i, batch_images in enumerate(loaded_data):
        # Generate noisy images
        noisy_batch = gen_y(batch_images)

        # Stack the noisy images into a single array
        noisy_images = [noisy_image for _, noisy_image in noisy_batch]

        # Save the noisy images as numpy files (y_batchIndex.npy)
        convert_to_numpy(noisy_images, os.path.join(output_dir, f"y_batch_{i}.npy"))

if __name__ == "__main__":
    main()

