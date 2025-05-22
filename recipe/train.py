import torch.nn as nn
from loguru import logger
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from typing import Optional
import os


class Trainer:
    """
    .fit() and .test() methods

    Args:
        model: torch model
        train_dataloader: contains training data as torch DataLoader
        test_dataloader: contains test data as torch DataLoader
        num_train_epochs: number of training epochs
        max_train_batches: optional argument specifying the maximum number of
            training batches after which to stop training
        logs_dir: directory to store model checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer_type: str = "sgd",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        num_train_epochs: int = 10,
        max_train_batches: Optional[int] = None,
        logs_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_train_epochs
        self.max_train_batches = max_train_batches
        self.logs_dir = logs_dir
        self.device = device

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()

    def get_optimizer(self) -> optim.Optimizer:
        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
            return optimizer
        raise ValueError(f"{self.optimizer_type} not supported")

    def fit(self) -> None:
        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # send to model's device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                log_interval = len(self.train_dataloader) // 5
                if i % log_interval == 0:
                    logger.info(
                        f"[epoch {epoch + 1}, steps {i + 1:5d}] loss: {running_loss / log_interval:.3f}"
                    )
                    running_loss = 0.0
        self.save_checkpoint()
        logger.info("Finished Training")

    def save_checkpoint(self):
        torch.save(
            {
                "epoch": self.num_epochs,
                "model_state_dict": self.model.state_dict(),
            },
            os.path.join(self.logs_dir, "checkpoint.pt"),
        )

    def test(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data

                # send to model's device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logger.info(
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
        )
