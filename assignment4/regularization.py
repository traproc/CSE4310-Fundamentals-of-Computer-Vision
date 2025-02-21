# Hien Dao
# CSE 4310 - Fundmentals of Computer Vision

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision import transforms
from torchvision.datasets import Imagenette
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

class BasicCNN_Regularization(pl.LightningModule):
    def __init__(self, num_classes=10):
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
            nn.Linear(2704, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, num_classes),
            nn.ReLU()
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)

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


# Prepare the dataset with data augmentation
train_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = Imagenette("data/imagenette/train/", split="train", size="160px", download=False, transform=train_transforms)

# Use 10% of the training set for validation
train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
val_dataset.dataset.transform = test_transforms

# Use DataLoader to load the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

# Configure the test dataset
test_dataset = Imagenette("data/imagenette/test/", split="val", size="160px", download=False, transform=test_transforms)

model = BasicCNN_Regularization()

# Add EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

# Configure Checkpoints
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

# Fit the model
trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], 
                     logger=pl_loggers.TensorBoardLogger('logs/BasicCNN_Regularization'), 
                     max_epochs=-1
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Evaluate the model on the test set
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=False)
trainer.test(model=model, dataloaders=test_loader)