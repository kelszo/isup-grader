#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer

from isupgrader.data.panda_dataset import preprocess_panda_database
import pandas as pd
import humanize


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s: %(message)s', datefmt="%d/%m %H:%M:%S")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("panda_path", help="Path to PANDA dataset", type=str)

    P = parser.parse_args()

    panda_path = Path(P.panda_path)

    df_raw = pd.read_csv(panda_path / "train.csv")

    logging.info("Starting: PANDA preprocessing database.")
    start = timer()

    df_processed = preprocess_panda_database(df_raw)

    df_processed.to_csv(panda_path / "train.processed.csv", index=False)

    end = timer()
    time_taken = humanize.naturaldelta(end - start)
    logging.info(f"Done processing. Preprocessing took {time_taken}")
