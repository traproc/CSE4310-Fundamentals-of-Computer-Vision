# Hien Dao
# CSE 4310 - Fundmentals of Computer Vision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import Imagenette
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

class BasicCNN_Regularization(pl.LightningModule):
    def __init__(self, num_features=2704, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.estimator = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, num_classes)
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return self.estimator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("val_accuracy", self.accuracy)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("test_accuracy", self.accuracy)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

imagenette_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cifar10_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Initialize the Imagenette dataset
imagenette_dataset = Imagenette("data/imagenette/train/", split="train", size="160px", download=False, transform=imagenette_transforms)

# Split Imagenette dataset into train and validation sets
train_size = int(0.9 * len(imagenette_dataset))
val_size = len(imagenette_dataset) - train_size
seed = torch.Generator().manual_seed(42)
imagenette_train_dataset, imagenette_val_dataset = torch.utils.data.random_split(imagenette_dataset, [train_size, val_size], generator=seed)
imagenette_val_dataset.dataset.transform = test_transforms

# Initialize DataLoader for Imagenette train and validation sets
imagenette_train_loader = torch.utils.data.DataLoader(imagenette_train_dataset, batch_size=128, num_workers=8, shuffle=True)
imagenette_val_loader = torch.utils.data.DataLoader(imagenette_val_dataset, batch_size=128, num_workers=8, shuffle=False)

# Train the model on Imagenette dataset
model = BasicCNN_Regularization()
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], max_epochs=-1, logger=pl_loggers.TensorBoardLogger('logs/transfer_learning'))
trainer.fit(model=model, train_dataloaders=imagenette_train_loader, val_dataloaders=imagenette_val_loader)

# Load the best model trained on Imagenette
trained_model = BasicCNN_Regularization.load_from_checkpoint(checkpoint_callback.best_model_path)

# Fine-tune the model on CIFAR-10 dataset
cifar10_dataset = CIFAR10("data/cifar10/", download=False, transform=cifar10_transforms)

train_size = int(len(cifar10_dataset) * 0.9)
val_size = len(cifar10_dataset) - train_size

cifar10_train_dataset, cifar10_val_dataset = torch.utils.data.random_split(cifar10_dataset, [train_size, val_size], generator=seed)
cifar10_val_dataset.dataset.transform = test_transforms

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=128, num_workers=8, shuffle=True)
cifar10_val_loader = torch.utils.data.DataLoader(cifar10_val_dataset, batch_size=128, num_workers=8, shuffle=False)

early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

fine_tuner = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], max_epochs=-1, logger=pl_loggers.TensorBoardLogger('logs/transfer_learning'))
fine_tuner.fit(model=trained_model, train_dataloaders=cifar10_train_loader, val_dataloaders=cifar10_val_loader)

# Configure the test dataset
test_dataset = Imagenette("data/imagenette/test/", split="val", size="160px", download=False, transform=test_transforms)

# Evaluate the model on the test set
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=False)
fine_tuner.test(model=trained_model, dataloaders=test_loader)