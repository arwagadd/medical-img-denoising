# import os
# import cv2
# import numpy as np

# def load_data(path: str) -> list[np.ndarray] :
#     data = []
#     image_dir = os.path.join(path, "dbs", "raw", "archive", "GAN-Traning Images")

#     if not os.path.isdir(image_dir):
#         print(f"Directory {image_dir} does not exist")
#         return data
    
#     for file_name in os.listdir(image_dir):
#         file_path = os.path.join(image_dir, file_name)

#         image = cv2.imread(file_path)

#         if image is not None: # if loaded successfully 
#             data.append(image)

#     return data

# # Testing load_data
# raw_data_path = ""
# loaded_images = load_data(raw_data_path)
# print(f"Number of loaded images: {len(loaded_images)}")

# if loaded_images:
#     print(f"Shape of the first loaded image: {loaded_images[0].shape}")

# def unify_shape(X: list[np.ndarray]) -> np.ndarray :
#     max_shape = tuple(max (dim) for dim in zip(*[x.shape for x in X]))
#     unified_data = np.zeros((len(X),) + max_shape, dtype=X[0].dtype)
#     for i, x in enumerate(X):

#         pad_widths = [(0, max_dim - curr_dim) for max_dim, curr_dim in zip(max_shape, x.shape)]
        
#         padded_x = np.pad(x, pad_widths, mode='constant')
        
#         unified_data[i] = padded_x

#     return unified_data

# # Test unify_shape
# dataset_path = ""
# dataset = load_data(dataset_path)
# unified_dataset = unify_shape(dataset)
# print("Shape of the first image in the unified dataset ", unified_dataset[0].shape)
# print("Shape of the third image in the unified dataset ", unified_dataset[2].shape)

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

    start_idx = 0
    while start_idx < len(image_files):
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        batch_images = []
        for file_name in batch_files:
            file_path = os.path.join(image_dir, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                batch_images.append(image)

        start_idx = end_idx
        yield batch_images

def unify_shape(X: list[np.ndarray]) -> np.ndarray:
    max_height = max(x.shape[0] for x in X)
    max_width = max(x.shape[1] for x in X)
    max_channels = max(x.shape[2] for x in X)

    unified_data = np.zeros((len(X), max_height, max_width, max_channels), dtype=np.uint8)
    
    for i, x in enumerate(X):
        height, width, channels = x.shape
        unified_data[i, :height, :width, :channels] = x
    
    return unified_data

def gen_y(X: List[np.ndarray]) -> List[tuple[np.ndarray, np.ndarray]] :
    noisy_images = []
    for x in X:
        noisy_image = x + np.random.normal(loc=0, scale=27, size=x.shape)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) 
        noisy_images.append((x, noisy_image))
    return noisy_images


def save_data(X: np.ndarray, Y: np.ndarray, path: str, batch_index: int) :
    if not os.path.exists(path):
        print(f"Directory {path} does not exist")
        return
    
    # extracting noisy images from gen_y
    for i, (original_image, noisy_image) in enumerate(zip(X,Y)):
        cv2.imwrite(os.path.join(path, f"image_{batch_index}_{i}.jpg"), noisy_image)


def main() :
    # Test the load_data and unify_shape
    raw_data_path = ""  
    batch_size = 32 

    loaded_data = load_data(raw_data_path, batch_size)
    total_images = 0
    for i, batch_images in enumerate(loaded_data):
        total_images += len(batch_images)
        unified_batch = unify_shape(batch_images)
        print(f"Batch {i+1}: Shape {unified_batch.shape}")

        noisy_images = gen_y(unified_batch)

        # Extracting noisy images from gen_y
        noisy_images_array = np.array([noisy_image for _, noisy_image in noisy_images])

        save_data(unified_batch, noisy_images_array, "dbs/intermittent/noisy_images",i)

        save_data(unified_batch,unified_batch, "dbs/intermittent/unified_images",i)

        # for original_image, noisy_image in noisy_images:
        #     # Display or save the original and noisy images
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        #     plt.title('Original Image')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        #     plt.title('Noisy Image')
        #     plt.show()

    print("The total number of loaded images is ", total_images)

if __name__== "__main__":
    main()
