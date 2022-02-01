from typing import Dict, List, Tuple, Union
import numpy as np

# From https://www.kaggle.com/rftexas/better-image-tiles-removing-white-spaces


def compute_statistics(image: np.ndarray) -> Tuple[float, float, float]:
    """Compute statistics of tile

    Args:
        image (np.ndarray): tile to return statistics of

    Returns:
        Tuple[float, float, float]: ratio_white_pixels, green_concentration, blue_concentration
    """

    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height

    num_white_pixels = 0

    summed_matrix = np.sum(image, axis=-1)
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    num_white_pixels = np.count_nonzero(summed_matrix > 620)
    ratio_white_pixels = num_white_pixels / num_pixels

    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])

    return ratio_white_pixels, green_concentration, blue_concentration


def select_k_best_regions(regions: np.ndarray, k: int) -> np.ndarray:
    """Select k best regions from regions_container.

    Args:
        regions (np.ndarray): Is array of (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
        k (int): [description]

    Returns:
        np.ndarray: [description]
    """
    regions = [x for x in regions if x[3] > 180 and x[4] > 180]
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return np.array(k_best_regions)


def get_k_best_regions(coordinates: np.ndarray, image: np.ndarray, tile_size: int) -> np.ndarray:
    regions = np.empty((len(coordinates), tile_size, tile_size, 3), dtype=np.uint8)

    for i, (x, y, _, _, _) in enumerate(coordinates):
        x, y = int(x), int(y)
        regions[i] = image[x: x+tile_size, y: y+tile_size, :]

    return regions


def generate_tiles(image: np.ndarray,
                   tile_size: int,
                   stride: int,
                   k: int,
                   only_tiles: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generates tiles from an image by tiling the image with strides (overlaps)
    and selecting the k best.

    Args:
        image (np.ndarray): [description]
        tile_size (int): [description]
        stride (int): [description]
        k (int): [description]
        only_tiles (bool, optional): [description]. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: [description]
    """

    max_width, max_height = image.shape[0], image.shape[1]
    regions_container: list[tuple[int, int, float, float, float]] = []
    i = 0

    while tile_size + stride*i <= max_height:
        j = 0

        while tile_size + stride*j <= max_width:
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride

            tile = image[
                x_top_left_pixel: x_top_left_pixel + tile_size,
                y_top_left_pixel: y_top_left_pixel + tile_size,
                :
            ]

            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(tile)

            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels,
                            green_concentration, blue_concentration)
            regions_container.append(region_tuple)

            j += 1

        i += 1

    regions_container = np.array(regions_container)

    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    tiles = get_k_best_regions(k_best_region_coordinates, image, tile_size)

    if only_tiles:
        return tiles
    else:
        return image, k_best_region_coordinates, tiles
