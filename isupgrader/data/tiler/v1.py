import os
from collections import Counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def add_mask_to_plot(plt, mask_data, provider):
    vmin = 0
    vmax = 5
    cmap = None

    if provider == "radboud":
        cmap = matplotlib.colors.ListedColormap(["white", "gray", "green", "yellow", "orange", "red"])
        vmax = 5
    else:
        cmap = matplotlib.colors.ListedColormap(["white", "green", "red"])
        vmax = 2

    plt.imshow(np.asarray(mask_data)[:, :, 0], interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)


def plot_mask(mask_data, provider):
    plt.figure(figsize=(8, 8))
    add_mask_to_plot(plt, mask_data, provider)
    plt.axis("off")
    plt.show()


def pad_tile(img, tile_size, mask, row, column):
    x, y = img.size
    img = np.asarray(img)

    colour = (255, 255, 255)
    if mask:
        colour = (0, 0, 0)

    if x < tile_size:
        if column == 0:
            img = cv2.copyMakeBorder(img, 0, 0, tile_size-x, 0, cv2.BORDER_CONSTANT, value=colour)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, 0, tile_size-x, cv2.BORDER_CONSTANT, value=colour)

    if y < tile_size:
        if row == 0:
            img = cv2.copyMakeBorder(img, tile_size-y, 0, 0, 0, cv2.BORDER_CONSTANT, value=colour)
        else:
            img = cv2.copyMakeBorder(img, 0, tile_size-y, 0, 0, cv2.BORDER_CONSTANT, value=colour)

    return img


def get_most_gleason_pixels(gleason_count):
    for i, count in enumerate(gleason_count):
        if count[0] > 2:
            return i

    return None


def get_amount_of_beningn_pixels(gleason_count, provider):
    benign_threshold = 1

    if provider == "radboud":
        benign_threshold = 2

    pixels = 0

    for i, count in enumerate(gleason_count):
        if count[0] <= benign_threshold:
            pixels += count[1]

    return pixels


def generate_tiles(img_id, area_threshold=0.1, gleason_area_threshold=0.05, benign_area_threshold=0.05,
                   benign_to_gleason_threshold=0.1, tile_size=512, overlap=64, level=12, figsize=(0, 0), draw=False):
    provider = train_labels.loc[img_id].data_provider

    biopsy = DeepZoomGenerator(
        OpenSlide(os.path.join(data_dir, f"{img_id}.tiff")),
        tile_size=tile_size - overlap * 2, overlap=overlap)
    biopsy_mask = DeepZoomGenerator(
        OpenSlide(os.path.join(mask_dir, f"{img_id}_mask.tiff")),
        tile_size=tile_size - overlap * 2, overlap=overlap)

    nrows = biopsy.level_tiles[level][1]
    ncols = biopsy.level_tiles[level][0]

    fig, axs = (None, None)
    fig_mask, axs_mask = (None, None)
    ret = []

    if draw:
        if figsize == (0, 0):
            figsize = (ncols*2, nrows*2)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        fig_mask, axs_mask = plt.subplots(nrows, ncols, figsize=figsize)

    for row in range(nrows):
        for column in range(ncols):
            img = biopsy.get_tile(level, (column, row))
            mask = biopsy_mask.get_tile(level, (column, row))
            gleason_score = 0

            gleason_pixels = np.asarray(mask)[:, :, 0].flatten()
            gleason_pixels = gleason_pixels[gleason_pixels != 0]
            gleason_count = Counter(gleason_pixels).most_common()
            gleason_pixels = 0
            benign_pixels = 0

            if provider == "radboud":
                if len(gleason_count) > 0:
                    index = get_most_gleason_pixels(gleason_count)
                    if index is not None:
                        gleason_score = gleason_count[index][0]

                if gleason_score == 2 or gleason_score == 1:
                    gleason_score = 0

                for count in gleason_count:
                    if count[0] <= 2:
                        benign_pixels += count[1]
                    else:
                        gleason_pixels += count[1]
            else:
                for count in gleason_count:
                    if count[0] == 1:
                        benign_pixels = count[1]
                    if count[0] == 2:
                        gleason_pixels = count[1]

                if benign_pixels == 0:
                    gleason_score = int(train_labels.loc[img_id].gleason_score.split("+")[0])
                elif gleason_pixels/benign_pixels > 0.2:
                    gleason_score = int(train_labels.loc[img_id].gleason_score.split("+")[0])

            # Pad edges
            img = pad_tile(img, tile_size, False, row, column)
            mask = pad_tile(mask, tile_size, True, row, column)

            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(imgray, 230, 255, cv2.THRESH_BINARY_INV)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            display_img = np.copy(img)
            display_mask_img = np.copy(mask)

            area = 0
            if len(cnts) > 0:
                cv2.drawContours(display_img, cnts, -1, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
                for cnt in cnts:
                    area += cv2.contourArea(cnt)

            if ((area/(tile_size**2) < area_threshold) or
                (gleason_score > 0 and gleason_pixels/area < gleason_area_threshold) or
                (gleason_score == 0 and benign_pixels/area < benign_area_threshold) or
                    (gleason_score == 0 and gleason_pixels > 0 and benign_pixels/gleason_pixels < benign_to_gleason_threshold)):

                close = int(tile_size/3)
                far = int(tile_size*2/3)

                display_img = cv2.add(display_img, np.array([-75.0]))
                cv2.line(display_img, (close, close), (far, far), (255, 0, 0), 5)
                cv2.line(display_img, (close, far), (far, close), (255, 0, 0), 5)

                cv2.line(display_mask_img, (close, close), (far, far), (255, 0, 0), 5)
                cv2.line(display_mask_img, (close, far), (far, close), (255, 0, 0), 5)
            else:
                ret.append([img, mask, gleason_score])

            if draw:
                axs[row, column].imshow(display_img, aspect="auto")
                axs[row, column].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

                add_mask_to_plot(axs_mask[row, column], display_mask_img, provider)
                axs_mask[row, column].set_aspect("auto")
                axs_mask[row, column].tick_params(which="both", bottom=False, left=False,
                                                  labelbottom=False, labelleft=False)

    if draw:
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.show()

        fig_mask.subplots_adjust(hspace=0, wspace=0)
        fig_mask.show()

    return ret
