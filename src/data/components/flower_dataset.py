import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms


class FlowerCLIPDataset(Dataset):
    def __init__(
        self, data_dir: str, metadata_file_path: str, id2word_file_path: str
    ) -> None:
        self.data_dir = data_dir
        id2word = np.load(id2word_file_path)  # id: (id, word)

        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f).to_dict("list")
            self.captions = []
            for captions in metadata["Captions"]:
                self.captions.append(
                    [
                        " ".join([id2word[int(id)][1] for id in caption])
                        .replace("<PAD>", "")
                        .strip()
                        for caption in captions
                    ]
                )
            self.image_paths = [
                os.path.join(data_dir, image_path)
                for image_path in metadata["ImagePath"]
            ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        caption = np.random.choice(self.captions[idx])
        image = decode_image(self.image_paths[idx], "RGB")
        return image, caption


class FlowerDataset(Dataset):
    def __init__(
        self, data_dir: str, metadata_file_path: str, id2word_file_path: str
    ) -> None:
        self.data_dir = data_dir
        id2word = np.load(id2word_file_path)  # id: (id, word)

        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f).to_dict("list")
            self.captions = []
            for captions in metadata["Captions"]:
                self.captions.append(
                    [
                        " ".join([id2word[int(id)][1] for id in caption])
                        .replace("<PAD>", "")
                        .strip()
                        for caption in captions
                    ]
                )
            self.image_paths = [
                os.path.join(data_dir, image_path)
                for image_path in metadata["ImagePath"]
            ]

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(64 * 76 / 64)),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        caption = np.random.choice(self.captions[idx])
        image = decode_image(self.image_paths[idx], "RGB")
        image = self.transform(image)
        return image, caption


class TestFlowerDataset(Dataset):
    def __init__(
        self, data_dir: str, metadata_file_path: str, id2word_file_path: str
    ) -> None:
        self.data_dir = data_dir
        id2word = np.load(id2word_file_path)  # id: (id, word)

        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f).to_dict("list")
            self.captions = [
                " ".join([id2word[int(id)][1] for id in caption])
                .replace("<PAD>", "")
                .strip()
                for caption in metadata["Captions"]
            ]
            self.ids = metadata["ID"]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        return self.ids[idx], self.captions[idx]
