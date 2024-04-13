from tqdm import tqdm
import numpy as np
import cv2 as cv
import os


def load_data(path: str) -> list[np.ndarray]:
    """ """
    ### get the image paths ###
    img_paths: list[str] = os.listdir(path)
    img_paths = [os.path.join(path, img) for img in img_paths]

    ### ~~~ truncate the images ~~~ ###
    trunc: int = 300
    img_paths = img_paths[:trunc]

    ### load the images ###
    imgs: list[np.ndarray] = list(map(cv.imread, img_paths))

    return imgs


def clean(X: list[np.ndarray], img_shape: tuple[int]) -> np.ndarray:
    """ """
    ### resize the images ###
    new_X: list[np.ndarray] = []
    for img in X:
        try:
            new_X.append(cv.resize(img, img_shape))
        except Exception:
            ...

    return np.stack(new_X)


def gen_y(X: np.ndarray) -> np.ndarray:
    """ """
    ### init vars ###
    y: np.ndarray
    epsilon: np.ndarray = np.random.normal(loc=0, scale=29, size=X.shape)

    ### generate y ###
    y = X + epsilon

    ### clip the values ###
    y = np.clip(y, 0, 255).astype(np.uint8)

    return y


def main() -> int:
    """ """
    ### init vars ###
    in_path: str = "./dbs/raw/"
    img_shape: tuple[int] = (300, 300)

    ### load the data ###
    X: lit[np.ndarray] = load_data(in_path)

    ### clean the data ###
    X = clean(X, img_shape)

    ### generate the y values ###
    y = gen_y(X)

    print(X.shape)
    print(y.shape)

    return 0


if __name__ == "__main__":
    main()
#   __   _,_ /_ __,
# _(_/__(_/_/_)(_/(_
#  _/_
# (/
