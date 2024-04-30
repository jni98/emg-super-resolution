import pytorch_lightning as pl
import torch
from torch import nn


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.bn = nn.BatchNorm1d(self.n_features)

        self.conv1 = nn.Conv1d(in_channels = self.n_features, out_channels = 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        # Transpose convolution layer to upscale to 65 channels
        # Adjust kernel size and stride to match your specific scaling needs
        self.deconv = nn.ConvTranspose1d(in_channels=self.hidden_size, out_channels=65, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        #input shape: [batch, channels, seq_len]
        x = self.bn(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Reshape for LSTM to: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # Reshape for final result to [batch, channels, seq_len]
        lstm_out = x.transpose(1, 2)
        out = self.deconv(lstm_out)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss