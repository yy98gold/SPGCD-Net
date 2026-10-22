import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from util import randrot, randfilp


class CDDataset(Dataset):

    def __init__(
            self,
            root,
            split='train',
            crop_size=256):

        super().__init__()

        assert split in [
            'train',
            'val',
            'test'
        ]

        self.root = root
        self.split = split
        self.crop_size = crop_size

        self.dir_A = os.path.join(
            root,
            split,
            "A"
        )

        self.dir_B = os.path.join(
            root,
            split,
            "B"
        )

        self.dir_label = os.path.join(
            root,
            split,
            "label"
        )

        self.names = sorted(
            os.listdir(self.dir_A)
        )

        self.names = [
            x for x in self.names
            if x.endswith(
                (
                    ".png",
                    ".jpg",
                    ".bmp",
                    ".tif"
                )
            )
        ]

        print(
            f"{split} dataset size: {len(self.names)}"
        )

    def __len__(self):

        return len(self.names)

    def load_image(self, path):

        img = Image.open(path).convert("RGB")

        return img

    def load_label(self, path):

        mask = Image.open(path).convert("L")

        return mask

    def random_crop(self, img1, img2, mask):

        i, j, h, w = \
            torch.transforms.RandomCrop.get_params(
                img1,
                output_size=(
                    self.crop_size,
                    self.crop_size
                )
            )

        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return img1, img2, mask

    def __getitem__(self, idx):

        name = self.names[idx]

        path_A = os.path.join(
            self.dir_A,
            name
        )

        path_B = os.path.join(
            self.dir_B,
            name
        )

        path_label = os.path.join(
            self.dir_label,
            name
        )

        img_A = self.load_image(path_A)
        img_B = self.load_image(path_B)
        label = self.load_label(path_label)

        ##################################################
        # Train Augmentation
        ##################################################

        if self.split == "train":

            if min(img_A.size) > self.crop_size:

                i, j, h, w = \
                    torch.randint(
                        0,
                        1,
                        (1,)
                    ), 0, 0, 0

                crop = torch.nn.Sequential()

                i, j, h, w = \
                    torch.utils.data.random_split(
                        [0],
                        [1]
                    )[0], 0, 0, 0

                i, j, h, w = TF.RandomCrop.get_params(
                    img_A,
                    output_size=(
                        self.crop_size,
                        self.crop_size
                    )
                )

                img_A = TF.crop(
                    img_A, i, j, h, w
                )

                img_B = TF.crop(
                    img_B, i, j, h, w
                )

                label = TF.crop(
                    label, i, j, h, w
                )

            if random.random() < 0.5:

                img_A = TF.hflip(img_A)
                img_B = TF.hflip(img_B)
                label = TF.hflip(label)

            if random.random() < 0.5:

                img_A = TF.vflip(img_A)
                img_B = TF.vflip(img_B)
                label = TF.vflip(label)

            rot = random.randint(0, 3)

            img_A = TF.rotate(
                img_A,
                90 * rot
            )

            img_B = TF.rotate(
                img_B,
                90 * rot
            )

            label = TF.rotate(
                label,
                90 * rot
            )

        ##################################################
        # Tensor
        ##################################################

        img_A = TF.to_tensor(img_A)

        img_B = TF.to_tensor(img_B)

        label = TF.to_tensor(label)

        ##################################################
        # Binary mask
        ##################################################

        label = (
            label > 0.5
        ).float()

        return (
            img_A,
            img_B,
            label,
            name        
        )