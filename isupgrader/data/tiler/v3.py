import numpy as np
import cv2


def _mask_tissue(image: np.ndarray, kernel_size: tuple = (7, 7), gray_threshold: int = 220) -> np.ndarray:
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask


def generate_tiles(img: np.ndarray, tile_size: int, min_tissue: float = 0.35, n_tiles: int = None):
    shape = img.shape
    pad0, pad1 = (tile_size - shape[0] %
                  tile_size) % tile_size, (tile_size - shape[1] % tile_size) % tile_size
    img = np.pad(img, [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]], constant_values=255)
    img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    if n_tiles is not None:
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
        img = img[idxs]
        return img
    else:
        tiles = []

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:256]

        img = img[idxs]

        for tile in img:
            if _mask_tissue(tile).mean() > min_tissue:
                tiles.append(tile)
            else:
                break

        return np.array(tiles)
