#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer

import cv2
import humanize
import isupgrader.data.tiler.v4
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from pandas.core.frame import DataFrame
from tqdm import tqdm
from isupgrader.utils.tqdm_joblib import tqdm_joblib

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s: %(message)s', datefmt="%d/%m %H:%M:%S")


def tile_wrap(path_to_slide: str, slide_id: str, tile_size: int, out_dir: str, level: int) -> DataFrame:
    image = tifffile.imread(path_to_slide, key=level)
    tiles = isupgrader.data.tiler.v4.generate_tiles(image, tile_size)

    # create output dir for slides
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # output tiles in order of info (high -> low). I.e. 0.png has most info
    for i, tile in enumerate(tiles):
        tile_out_path = os.path.join(out_dir, f"{i}.png")
        cv2.imwrite(tile_out_path, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))

    df = pd.DataFrame([{"slide_id": slide_id, "n_tiles": len(tiles)}])

    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("panda_path", help="Path to PANDA dataset", type=str)
    parser.add_argument("output_path", help="Path to output dataset", type=str)
    parser.add_argument("--slide-level", help="Number of parallel processes", type=int, default=1)
    parser.add_argument("--threads", help="Number of parallel processes", type=int, default=4)
    parser.add_argument("--tile-size", help="Width/height of output tiles in pixels", type=int, default=256)
    parser.add_argument('--debug', help="Debug", dest='debug', action='store_true', default=False)

    P = parser.parse_args()

    panda_path = P.panda_path
    output_path = P.output_path

    panda_df_path = os.path.join(panda_path, "train.processed.csv")
    df = pd.read_csv(panda_df_path)

    logging.info("Starting: PANDA Tiling.")
    logging.info(f"panda_path={P.panda_path}")
    logging.info(f"output_path={P.output_path}")
    logging.info(f"slide_level={P.slide_level}")
    logging.info(f"threads={P.threads}")
    logging.info(f"tile_size={P.tile_size}")

    start = timer()

    if P.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("~~DEBUGGING~~")
        df_out = tile_wrap(path_to_slide=os.path.join(panda_path, "train_images", f"{df.iloc[1].slide_id}.tiff"),
                           slide_id=df.iloc[0].slide_id,
                           tile_size=P.tile_size,
                           out_dir=os.path.join(output_path, "train_images", df.iloc[1].slide_id),
                           level=P.slide_level)

        df_slides = [df_out]
    else:
        with tqdm_joblib(tqdm(total=len(df))) as progress_bar:
            df_slides = Parallel(n_jobs=P.threads)(delayed(tile_wrap)(
                path_to_slide=os.path.join(panda_path, "train_images", f"{df.iloc[i].slide_id}.tiff"),
                slide_id=df.iloc[i].slide_id,
                tile_size=P.tile_size,
                out_dir=os.path.join(output_path, "train_images", df.iloc[i].slide_id),
                level=P.slide_level
            ) for i in range(len(df)))

    df_slides = pd.concat(df_slides)

    df_out = pd.merge(df, df_slides, on="slide_id")

    df_out.to_csv(os.path.join(output_path, "slides_train.csv"), index=False)

    end = timer()

    time_taken = humanize.naturaldelta(end - start)

    n_slides = df_out["n_tiles"].notnull().sum()
    time_per_slide = (end - start) / n_slides

    total_panda_size = len(pd.read_csv(os.path.join(panda_path, "train_complete.csv")))

    logging.info("Done tiling")
    logging.info("-----------")
    logging.info(f"Total time: {time_taken}")
    logging.info(f"Total slides: {n_slides}")
    logging.info(f"Time per slide: {time_per_slide:.2f}s")

    if n_slides < total_panda_size:
        panda_time_estimation = humanize.naturaldelta(time_per_slide * total_panda_size)
        logging.info(f"Estimated time to process complete PANDA dataset: {panda_time_estimation}")
