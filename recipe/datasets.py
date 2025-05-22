import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path


USER = os.environ.get("USER")


class CIFAR10:
    def __init__(
        self, root: str = f"/checkpoint/{USER}/datasets/", batch_size: int = 4
    ):
        self.root = root
        # make dataset directory if it doesn't exist
        Path(self.root).mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def get_trainloader(self) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=self.transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        return trainloader

    def get_testloader(self) -> DataLoader:
        testset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        return testloader
