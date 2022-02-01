import os
from typing import Tuple

import albumentations
import cv2
import isupgrader.data.tiler.v4
import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset


def preprocess_panda_database(df: DataFrame) -> DataFrame:
    # change gleason negative to 0+0
    df["gleason_score"] = df["gleason_score"].apply(lambda x: "0+0" if x == "negative" else x)

    # drop incorrectly graded slide (gleason does not correspond ISUP)
    df = df[df.image_id != "b0a92a74cb53899311acc30b7405e101"]

    # rename image_id to slide_id for logical reasons
    df = df.rename(columns={"image_id": "slide_id"})

    return df


class ImageTransform():
    def __init__(self,
                 stage: str,
                 tile_size: int,
                 ):
        self.stage = stage

        self.data_transform = {
            'TRAIN:TILE': albumentations.Compose([
                albumentations.Transpose(),
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip(),
                albumentations.RandomRotate90(),
                albumentations.Blur(blur_limit=2),
                albumentations.ColorJitter(brightness=0.2, contrast=0.2,
                                           saturation=0.03, hue=0.02),
                albumentations.Resize(tile_size, tile_size, interpolation=cv2.INTER_LANCZOS4),
                albumentations.CoarseDropout(max_holes=1, min_holes=1, max_height=tile_size//3, max_width=tile_size//3,
                                             fill_value=0),
            ]),
            'VAL:TILE': albumentations.Compose([
                albumentations.Resize(tile_size, tile_size, interpolation=cv2.INTER_LANCZOS4),
            ]),
            'TEST:TILE': albumentations.Compose([
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip(),
                albumentations.RandomRotate90(),
                albumentations.Resize(tile_size, tile_size, interpolation=cv2.INTER_LANCZOS4),
            ]),

            'TRAIN:SLIDE': albumentations.Compose([
                albumentations.Transpose(),
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip(),
                albumentations.Normalize()
            ]),
            'VAL:SLIDE': albumentations.Compose([
                albumentations.Normalize()
            ]),
            'TEST:SLIDE': albumentations.Compose([
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip(),
                albumentations.RandomRotate90(),
                albumentations.Normalize()
            ]),
        }

    def __call__(self, img: np.ndarray, level: str) -> np.ndarray:
        return self.data_transform[f"{self.stage}:{level}"](image=img)['image']


class PANDADataset(Dataset):
    def __init__(self,
                 df: DataFrame,
                 slide_dir: str,
                 n_tiles: int,
                 tile_size: int,
                 transform: ImageTransform,
                 live_tile: bool = False,
                 live_tile_level: int = 1,
                 live_tile_size: int = 256,
                 ):

        self.df = df.reset_index(drop=True)
        self.slide_dir = slide_dir

        # live tile
        self.live_tile = live_tile
        self.live_tile_level = live_tile_level
        self.live_tile_size = live_tile_size

        # Tiling
        self.n_tiles = n_tiles
        self.tile_size = tile_size

        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        slide_data = self.df.iloc[index]

        tiles = np.full((self.n_tiles, self.tile_size, self.tile_size, 3), fill_value=255, dtype=np.uint8)

        if self.live_tile:
            slide_path = os.path.join(self.slide_dir, f"{slide_data.slide_id}.tiff")
            img = tifffile.imread(slide_path, key=self.live_tile_level)
            generated_tiles = isupgrader.data.tiler.v4.generate_tiles(img, self.live_tile_size)
            del img

            if len(generated_tiles) > self.n_tiles:
                np.random.shuffle(generated_tiles)
                generated_tiles = generated_tiles[:self.n_tiles]

            for i, tile in enumerate(generated_tiles):
                tiles[i] = cv2.resize(tile, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LANCZOS4)
        else:
            slide_path = os.path.join(self.slide_dir, slide_data.slide_id)

            # Get all tile for a slide and pick n_tiles worth of tiles
            tile_paths = os.listdir(slide_path)

            if len(tile_paths) > self.n_tiles:
                tile_paths = np.random.choice(tile_paths, self.n_tiles, replace=False)

            for i, tile_path in enumerate(tile_paths):
                tile_path = os.path.join(slide_path, tile_path)
                tiles[i] = cv2.cvtColor(
                    cv2.resize(cv2.imread(tile_path),
                               (self.tile_size, self.tile_size), interpolation=cv2.INTER_LANCZOS4),
                    cv2.COLOR_BGR2RGB)

        np.random.shuffle(tiles)

        # Create a glued image with sqrt(n_tiles) x sqrt(n_tiles) dimension
        n_row_tiles = int(np.sqrt(self.n_tiles))
        slide = np.empty((self.tile_size * n_row_tiles, self.tile_size * n_row_tiles, 3), dtype=np.uint8)

        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                # apply tile level augmentations
                tile = self.transform(tiles[i], "TILE")

                h1 = h * self.tile_size
                w1 = w * self.tile_size
                slide[h1:h1+self.tile_size, w1:w1+self.tile_size] = tile

        # apply slide (glued image) level augmentations
        slide = self.transform(slide, "SLIDE")

        # Only if not normalising
        #slide = slide.astype(np.float32)
        #slide /= 255

        # network needs channel first
        slide = slide.transpose(2, 0, 1)

        return torch.from_numpy(slide), torch.tensor([slide_data.isup_grade], dtype=torch.float32), slide_data.slide_id


class PANDADataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: DataFrame,
                 test_df: DataFrame,
                 train_slide_dir: str,
                 test_slide_dir: str,
                 n_tiles: int,
                 tile_size: int,
                 batch_size: int,
                 num_workers: int,
                 fold: int,
                 ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.train_slide_dir = train_slide_dir
        self.test_slide_dir = test_slide_dir
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold

    def setup(self, stage: str):
        if stage == "fit":
            self.PANDA_train = PANDADataset(df=self.train_df[(self.train_df['fold'] != self.fold)],
                                            slide_dir=self.train_slide_dir,
                                            n_tiles=self.n_tiles,
                                            tile_size=self.tile_size,
                                            transform=ImageTransform(stage="TRAIN", tile_size=self.tile_size)
                                            )

            self.PANDA_val = PANDADataset(df=self.train_df[(self.train_df['fold'] == self.fold)],
                                          slide_dir=self.train_slide_dir,
                                          n_tiles=self.n_tiles,
                                          tile_size=self.tile_size,
                                          transform=ImageTransform(stage="VAL", tile_size=self.tile_size)
                                          )
        elif stage == "test":
            self.PANDA_test = PANDADataset(df=self.test_df,
                                           slide_dir=self.test_slide_dir,
                                           n_tiles=self.n_tiles,
                                           tile_size=self.tile_size,
                                           transform=ImageTransform(stage="TEST", tile_size=self.tile_size),
                                           live_tile=True,
                                           live_tile_level=1,
                                           live_tile_size=256,
                                           )
        else:
            raise Exception("No such stage", stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.PANDA_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.PANDA_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.PANDA_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
