import logging
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import humanize
import pandas as pd
import torch.nn as nn
from isupgrader.data.panda_dataset import PANDADataModule
from isupgrader.models.isup_grader_model import ISUPGraderModel
from isupgrader.networks.efficientnet import Enet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s: %(message)s', datefmt="%d/%m %H:%M:%S")
logger = logging.getLogger("pytorch_lightning")

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--processed-panda-path", type=str, default="/data/processed/panda")
    parser.add_argument("--raw-panda-path", type=str, default="/data/raw/panda")
    parser.add_argument("--out-path", type=str, default="/app/out")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--n-tiles", type=int, default=36)

    parser.add_argument("--n-workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', default=False)
    parser.add_argument('--save', dest='save', action='store_true', default=False)
    parser.add_argument("--ex-name", type=str)

    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--debug-sample-size',  type=int, default=64)

    P = parser.parse_args()

    OUT_PATH = Path(P.out_path)
    logger.addHandler(logging.FileHandler(OUT_PATH / "isup-grader.log"))

    PROCESSED_PANDA_DATASET_PATH = Path(P.processed_panda_path)
    RAW_PANDA_DATASET_PATH = Path(P.raw_panda_path)

    TRAIN_IMAGES_PATH = PROCESSED_PANDA_DATASET_PATH / "train_images/"
    df_train = pd.read_csv(PROCESSED_PANDA_DATASET_PATH / "slides_train.csv")

    TEST_SLIDES_PATH = RAW_PANDA_DATASET_PATH / "test_images/"
    df_test = pd.read_csv(RAW_PANDA_DATASET_PATH/"test.csv")
    df_derived = pd.read_csv(RAW_PANDA_DATASET_PATH/"derived.csv")
    df_test = pd.merge(df_test, df_derived, on="image_id")
    df_test = df_derived.rename(columns={"image_id": "slide_id"})

    if P.debug:
        P.max_epochs = 2

        if len(df_train) < P.debug_sample_size:
            P.debug_sample_size = len(df_train)

        df_train = df_train.sample(P.debug_sample_size).reset_index(drop=True)
        df_test = df_test.sample(P.debug_sample_size//4).reset_index(drop=True)


    seed_everything(P.seed, workers=True)

    logging.info("Starting: Training isup-grader on PANDA dataset.")
    start = timer()

    callbacks = []

    if P.save:
        logging.info("Activated saving.")

        if P.ex_name:
            filename = P.ex_name
        else:
            date = datetime.now().strftime("%Y%m%d%H")
            filename = f"{date}-{P.fold}-" + '{val_epoch_qwk:.3f}'

        checkpoint_callback = ModelCheckpoint(dirpath=OUT_PATH / 'checkpoints', filename=filename, monitor='val_epoch_loss', save_top_k=1, mode='min')
        callbacks.append(checkpoint_callback)

    if P.early_stopping:
        logging.info("Activated early stopping.")
        callbacks.append(EarlyStopping(monitor='val_epoch_loss', min_delta=0.0, patience=5))

    trainer = Trainer(
        max_epochs=P.max_epochs,
        callbacks=callbacks,
        deterministic=True,
        devices=-1,
        accelerator='gpu',
        plugins=DDPPlugin(find_unused_parameters=False),
        logger=TensorBoardLogger("logs", name=os.path.join(OUT_PATH, "isup-grader")),
        sync_batchnorm=True,
        precision=16,
        benchmark=True,
    )

    panda = PANDADataModule(train_df=df_train,
                            test_df=df_test,
                            train_slide_dir=TRAIN_IMAGES_PATH,
                            test_slide_dir=TEST_SLIDES_PATH,
                            n_tiles=P.n_tiles,
                            tile_size=P.tile_size,
                            batch_size=P.batch_size,
                            num_workers=P.n_workers,
                            fold=P.fold)

    enet = Enet(1)

    model = ISUPGraderModel(net=enet,
                            criterion=nn.MSELoss(),
                            lr=P.learning_rate,
                            weight_decay=P.weight_decay,
                            n_epochs=P.max_epochs,
                            datamodule=panda)

    trainer.fit(model, panda)

    panda.setup("test")
    trainer.test(model, dataloaders=panda.test_dataloader())

    end = timer()
    time_taken = humanize.naturaldelta(end - start)
    logging.info(f"Done training. Training took {time_taken}")
