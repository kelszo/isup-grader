from pathlib import Path
from glob import glob
from argparse import ArgumentParser

import pandas as pd
import torch.nn as nn
from isupgrader.data.panda_dataset import PANDADataModule
from isupgrader.models.isup_grader_model import ISUPGraderModel
from isupgrader.networks.efficientnet import Enet
from pytorch_lightning import Trainer, seed_everything
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("version", type=str)
    parser.add_argument("checkpoint", type=str)

    parser.add_argument("--tta", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="internal", choices=["internal", "external"])

    parser.add_argument("--processed-panda-path", type=str, default="/data/processed/panda")
    parser.add_argument("--raw-panda-path", type=str, default="/data/raw/panda")
    parser.add_argument("--out-path", type=str, default="/app/out")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--n-tiles", type=int, default=36)

    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)

    P = parser.parse_args()

    seed_everything(P.seed, workers=True)

    if P.dataset == "internal":
        RAW_PANDA_DATASET_PATH = Path("/data/raw/panda")
        TEST_SLIDES_PATH = RAW_PANDA_DATASET_PATH / "test_images/"
        df_test = pd.read_csv(RAW_PANDA_DATASET_PATH/"test.csv")
        df_derived = pd.read_csv(RAW_PANDA_DATASET_PATH/"derived.csv")

        df_test = pd.merge(df_test, df_derived, on="image_id")
        df_test = df_derived.rename(columns={"image_id": "slide_id"})
    else:
        RAW_PANDA_EXTERNAL_PATH = Path("/data/raw/panda_external_ks")
        TEST_SLIDES_PATH = RAW_PANDA_EXTERNAL_PATH / "test_images"
        df_test = pd.read_csv(RAW_PANDA_EXTERNAL_PATH / "pecan_external_ks_labels_20200724.csv")

        df_test['ISUP'] = df_test['ISUP'].apply(lambda x: int(x[-1]))

        df_test = df_test.rename(columns={"image": "slide_id", "ISUP": "isup_grade"})
        df_test = df_test[['slide_id', 'isup_grade']]

    df_test = df_test.sort_values("slide_id")

    df_test["isup_grade_pred"] = -1

    enet = Enet(5)

    model = ISUPGraderModel.load_from_checkpoint(P.checkpoint, net=enet)

    trainer = Trainer(
        deterministic=True,
        gpus=1,
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

    preds = []
    preds_raw = []

    for i in range(1, P.tta + 1):
        print(f"TTA: {i}/{P.tta}")
        trainer.test(model, panda)

        for sub in glob('/tmp/predictions/*.csv'):
            preds.append(pd.read_csv(sub).sort_values("slide_id").isup_grade_pred.to_numpy())
            preds_raw.append(pd.read_csv(sub).sort_values("slide_id").isup_grade_pred_raw.to_numpy())

    final_preds = []
    final_preds_majority = []
    final_preds_mean = []

    for pred, pred_raw in zip(zip(*preds), zip(*preds_raw)):
        final_preds_majority.append(np.bincount(pred).argmax())
        final_preds_mean.append(round(np.mean(pred_raw)))

        voting_count = np.bincount(pred)
        top_isup = voting_count.argmax()

        voting_count[top_isup] = 0
        second_isup = voting_count.argmax()

        voting_count = np.bincount(pred)

        if voting_count[top_isup] > voting_count[second_isup]:
            final_preds.append(np.bincount(pred).argmax())
        else:
            final_preds.append(round(np.mean(pred_raw)))

    df_test.isup_grade_pred = final_preds
    df_test.to_csv(f"/app/out/{P.version}-{P.dataset}-predictions.csv")

    df_test.isup_grade_pred = final_preds_mean
    df_test.to_csv(f"/app/out/{P.version}-{P.dataset}-predictions_mean.csv")

    df_test.isup_grade_pred = final_preds_majority
    df_test.to_csv(f"/app/out/{P.version}-{P.dataset}-predictions_majority.csv")
