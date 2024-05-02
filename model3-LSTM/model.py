import os

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import plotly.graph_objects as go
import wandb


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
                 kernel_size,
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
        self.kernel_size = kernel_size

        if self.kernel_size == 3:
            self.padding = 1
        elif self.kernel_size == 5:
            self.padding = 2

        self.bn = nn.BatchNorm1d(self.n_features)

        self.conv1 = nn.Conv1d(in_channels = self.n_features, out_channels = 64, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=self.kernel_size, padding=self.padding)

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
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        # Log the first validation batch for visualization
        if batch_idx == 0:
            self.validation_plots(x,y,y_hat, 'Validation')

        return loss

    def test_step(self, batch, batch_idx):
        x, y, y_path = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        # Log the first test batch for visualization
        if batch_idx == 0:
            self.validation_plots(x,y,y_hat, 'Test')

        # Save the prediction for each batch individually
        results_dir = self.trainer.logger.save_dir  # or any other directory you have defined
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        y_path = y_path[0].replace("hr", "sr")
        # File name with batch index
        file_path = os.path.join(results_dir, y_path)
        np.save(file_path, y_hat.cpu().numpy())  # Save as NumPy array

        print(f"Saved batch {batch_idx} predictions to {file_path}")

        return loss

    def validation_plots(self, x, y, y_hat, phase):

        # Convert tensors to numpy arrays and move to CPU
        #val_lr_destandardized = np.asarray(x.to('cpu'))
        # TODO: Destandardize
        val_hr_destandardized = np.asarray(y.to('cpu'))
        val_sr_destandardized = np.asarray(y_hat.to('cpu'))

        # Extract specific channels
        #lr_channel = val_lr_destandardized[0][6][2]
        val_channel = val_hr_destandardized[0][3]
        sr_channel = val_sr_destandardized[0][3]

        ############################## Log Plots given channel########################################
        print("Logging plots for batch index 0 given channel")

        # x-axis for the following logs
        bar_fig_1, axs = plt.subplots(3, 1, figsize=(8, 10))

        # Log one channel in the middle of the grid
        axs[0].plot(val_channel)
        axs[0].set_title('Validation channel [6][2]')

        # Log one channel in the middle of the grid
        axs[1].plot(sr_channel)
        axs[1].set_title('Super-resolution channel [6][2]')

        # log cross correlation
        auto_correlation = np.correlate(val_channel, val_channel, mode='full')
        cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        axs[2].plot(cross_correlation)
        axs[2].set_title('Cross Correlation channel [6][2]')

        plt.tight_layout()
        wandb.log({phase + " given Channel": [wandb.Image(bar_fig_1)]})
        # Close the figure
        plt.close(bar_fig_1)

        # New figure for plots of channels
        # Create Plotly figure
        plotly_fig = go.Figure()
        # Add traces for all channels
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution channel [6][2]'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(val_channel)), y=val_channel, name='Validation channel [6][2]', opacity=0.5))
        #plotly_fig.add_trace(
            #go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][2]', opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - given Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")
        # Log the Plotly figure to W&B
        wandb.log({phase + " - SR vs Val - given Channel": plotly_fig})

        ############# Plot frequency histogram# #################
        sampling_freq = 2048

        # Compute FFT for val_channel
        n = len(val_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T  # Two sides frequency range
        frq = frq[range(n // 2)]  # One side frequency range
        val_Y = np.fft.fft(val_channel) / n
        val_Y = val_Y[range(n // 2)]

        # Compute FFT for sr_channel
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]

        # Compute FFT for lr_channel
        #lr_Y = np.fft.fft(lr_channel) / n
        #lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        #bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

        # Plot sr_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution channel', opacity=1, offset=0.1))

        bar_fig_1.update_layout(title='Frequency Spectrum - Given channel',
                                xaxis_title='Frequency (Hz)',
                                yaxis_title='Magnitude',
                                barmode='group')

        # Log figure to wandb
        wandb.log({phase + " - Frequency Spectrum - Given channel": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(val_Y), mode='lines', name='Validation (HR) channel', opacity=0.5))

        # Plot lr_channel
        #freq_spec1.add_trace(
        #    go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution channel', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution channel', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Given channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({phase + " - Frequency Spectrum - Given channel": freq_spec1})

        ############################## Log Plots predicted channel########################################
        print("Logging plots for batch index 0 predicted channel")

        # Select channels
        val_channel = val_hr_destandardized[0][2]
        sr_channel = val_sr_destandardized[0][2]
        #lr_channel = val_lr_destandardized[0][2][6][1]

        # x-axis for the following logs
        fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10))

        # Log one channel in the middle of the grid
        axs1[0].plot(val_channel)
        axs1[0].set_title('Validation channel [6][1]')

        # Log one channel in the middle of the grid
        axs1[1].plot(sr_channel)
        axs1[1].set_title('Super-resolution channel [6][1]')

        # log cross correlation
        auto_correlation = np.correlate(val_channel, val_channel, mode='full')
        cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        axs1[2].plot(cross_correlation)
        axs1[2].set_title('Cross Correlation channel [6][1]')

        plt.tight_layout()
        wandb.log({phase + " predicted Channel": [wandb.Image(fig1)]})

        # Close the figure
        plt.close(fig1)

        # New figure for plots of channels
        # Create Plotly figure
        plotly_fig = go.Figure()
        # Add traces for all channels
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution channel [6][1]'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(val_channel)), y=val_channel, name='Validation channel [6][1]', opacity=0.5))
        #plotly_fig.add_trace(
            #go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][1]', opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - predicted Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")

        # Log the Plotly figure to W&B
        wandb.log({phase + " - SR vs Val - predicted Channel": plotly_fig})

        ############# Plot frequency histogram# #################
        # Generate sample data for demonstration purposes
        sampling_freq = 2048

        # Compute FFT for val_channel
        n = len(val_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T  # Two sides frequency range
        frq = frq[range(n // 2)]  # One side frequency range
        val_Y = np.fft.fft(val_channel) / n
        val_Y = val_Y[range(n // 2)]

        # Compute FFT for sr_channel
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]

        # Compute FFT for lr_channel
        #lr_Y = np.fft.fft(lr_channel) / n
        #lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        #bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

        # Plot sr_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution channel', opacity=1, offset=0.1))

        bar_fig_1.update_layout(title='Frequency Spectrum - Predicted channel',
                                xaxis_title='Frequency (Hz)',
                                yaxis_title='Magnitude',
                                barmode='group')

        # Log figure to wandb
        wandb.log({phase + " - Frequency Spectrum - Predicted channel": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(val_Y), mode='lines', name='Validation (HR) channel', opacity=0.5))

        # Plot lr_channel
        #freq_spec1.add_trace(
        #    go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution channel', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution channel', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Predicted channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({phase + " - Frequency Spectrum - Predicted channel": freq_spec1})