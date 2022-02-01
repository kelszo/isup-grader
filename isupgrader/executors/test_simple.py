from pathlib import Path
from glob import glob
from argparse import ArgumentParser
import logging
import warnings
from timeit import default_timer as timer

import pandas as pd
import torch.nn as nn
from isupgrader.data.panda_dataset import PANDADataModule
from isupgrader.models.isup_grader_model import ISUPGraderModel
from isupgrader.networks.efficientnet import Enet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
import humanize

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s: %(message)s', datefmt="%d/%m %H:%M:%S")
logger = logging.getLogger("pytorch_lightning")

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("checkpoint", type=str)

    parser.add_argument("--dataset", type=str, default="internal", choices=["internal", "external"])

    parser.add_argument("--processed-panda-path", type=str, default="/data/processed/panda")
    parser.add_argument("--raw-panda-path", type=str, default="/data/raw/panda")
    parser.add_argument("--out-path", type=str, default="/app/out")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--n-tiles", type=int, default=36)

    parser.add_argument("--n-workers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)

    P = parser.parse_args()

    seed_everything(P.seed, workers=True)

    logging.info("Starting: testing isup-grader.")
    start = timer()


    if P.dataset == "internal":
        logging.info("Using: Internal dataset.")
        RAW_PANDA_DATASET_PATH = Path("/data/raw/panda")
        TEST_SLIDES_PATH = RAW_PANDA_DATASET_PATH / "test_images/"
        df_test = pd.read_csv(RAW_PANDA_DATASET_PATH/"test.csv")
        df_derived = pd.read_csv(RAW_PANDA_DATASET_PATH/"derived.csv")

        df_test = pd.merge(df_test, df_derived, on="image_id")
        df_test = df_derived.rename(columns={"image_id": "slide_id"})
    else:
        logging.info("Using: External dataset.")
        RAW_PANDA_EXTERNAL_PATH = Path("/data/raw/panda_external_ks")
        TEST_SLIDES_PATH = RAW_PANDA_EXTERNAL_PATH / "test_images"
        df_test = pd.read_csv(RAW_PANDA_EXTERNAL_PATH / "pecan_external_ks_labels_20200724.csv")

        df_test['ISUP'] = df_test['ISUP'].apply(lambda x: int(x[-1]))

        df_test = df_test.rename(columns={"image": "slide_id", "ISUP": "isup_grade"})
        df_test = df_test[['slide_id', 'isup_grade']]

    df_test = df_test.sort_values("slide_id")

    df_test["isup_grade_pred"] = -1

    enet = Enet(1)

    trainer = Trainer(
        deterministic=True,
        accelerator='gpu',
        devices=[0],
        #plugins=DDPPlugin(find_unused_parameters=False),
        precision=16,
        benchmark=True,
        default_root_dir="/app/out/logs"
    )

    panda = PANDADataModule(train_df=None,
                            test_df=df_test,
                            train_slide_dir=None,
                            test_slide_dir=TEST_SLIDES_PATH,
                            n_tiles=P.n_tiles,
                            tile_size=P.tile_size,
                            batch_size=P.batch_size,
                            num_workers=P.n_workers,
                            fold=1
                            )

    model = ISUPGraderModel.load_from_checkpoint(P.checkpoint, net=enet, datamodule=panda)

    panda.setup("test")
    trainer.test(model, dataloaders=panda.test_dataloader())

    end = timer()
    time_taken = humanize.naturaldelta(end - start)
    logging.info(f"Done testing. Testing took {time_taken}")