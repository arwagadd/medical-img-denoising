from tqdm import tqdm
import numpy as np
import cv2 as cv
import os


def load_data(path: str) -> list[np.ndarray]:
    """ """
    ### get all paths ###
    img_names: list[str] = os.listdir(path)
    img_paths: list[str] = [os.path.join(path, img_name) for img_name in img_names]
    img_paths = list(
        filter(lambda x: ".jpg" in x or ".png" in x or ".jpeg" in x, img_paths)
    )

    ### ~~~ truncate ~~~ ###
    img_paths = img_paths[:3000]

    ### init lambda functions ###
    def load_a_img(img_path: str) -> np.ndarray | None:
        try:
            return cv.imread(img_path)
        except Exception as e:
            return None

    ### load images ###
    imgz: list[np.ndarray | None] = list(map(load_a_img, img_paths))

    ### filter out None values ###
    filtered_imgz: list[np.ndarray] = list(filter(lambda x: x is not None, imgz))

    return filtered_imgz


def clean(X: list[np.ndarray], img_shape: tuple[int, int] = (300, 300)) -> np.ndarray:
    """ """
    ### unify the sizes ###
    X = list(map(lambda x: cv.resize(x, img_shape), X))

    ### convert to numpy array ###
    X = np.stack(X)

    return X


def gen_y(X: np.ndarray) -> np.ndarray:
    """ """
    y: list[np.ndarray] = []
    for img in tqdm(X):
        epsilon: np.ndarray = np.random.normal(loc=0, scale=29, size=img.shape)
        noisy_img: np.ndarray = img + epsilon
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        y.append(noisy_img)

    y: np.ndarray = np.stack(y)

    return y


def main() -> int:
    """ """
    ### init vars ###
    in_path: str = "./dbs/raw/"
    out_path: str = "./dbs/intermittent/"
    img_size: tuple[int, int] = (300, 300)

    ### load data ###
    X: list[np.ndarray] = load_data(in_path)

    ### clean data ###
    X: np.ndarray = clean(X, img_size)

    ### generate y ###
    y: np.ndarray = gen_y(X)

    ### switch the X and y ###
    X, y = y, X

    ### save the data ###
    np.save(os.path.join(out_path, "X.npy"), X)
    np.save(os.path.join(out_path, "y.npy"), y)

    return 0


if __name__ == "__main__":
    main()
#   __   _,_ /_ __,
# _(_/__(_/_/_)(_/(_
#  _/_
# (/
