'''
All parameters are aggregated in one place.
This is useful for reporting experiment params to experiment tracking software
'''
import torch
from torch import nn
from data import CustomDataModule
from model import LSTMRegressor
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="emg-super-resolution-LSTM")

#logger = TensorBoardLogger("logs", name="SR LSTM")


p = dict(
    seq_len = 200,
    batch_size = 1,
    criterion = nn.MSELoss(),
    max_epochs = 5,
    n_features = 21,
    hidden_size = 65,
    num_layers = 1,
    dropout = 0.2,
    learning_rate = 0.001,
)

seed_everything(1)

trainer = Trainer(
    max_epochs=p['max_epochs'],
    devices=1,
    accelerator="gpu",
    deterministic=True,
    logger=wandb_logger
)

model = LSTMRegressor(
    n_features = p['n_features'],
    hidden_size = p['hidden_size'],
    seq_len = p['seq_len'],
    batch_size = p['batch_size'],
    criterion = p['criterion'],
    num_layers = p['num_layers'],
    dropout = p['dropout'],
    learning_rate = p['learning_rate']
)

dm = CustomDataModule(
    seq_len = p['seq_len'],
    batch_size = p['batch_size']
)

trainer.fit(model, dm)
trainer.test(model, datamodule=dm)