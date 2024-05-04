"""
Model used with sequence length 10240 compressed to sequence length 160 ( --> c = 64)
"""
import os

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import plotly.graph_objects as go
import wandb


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()

        self.conv2 = nn.Conv3d(1, 16, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn2 = nn.BatchNorm3d(16)
        self.res2 = ResidualBlock3D(16)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn3 = nn.BatchNorm3d(32)
        self.res3 = ResidualBlock3D(32)

        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn4 = nn.BatchNorm3d(64)
        self.res4 = ResidualBlock3D(64)

        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn5 = nn.BatchNorm3d(64)
        self.res5 = ResidualBlock3D(64)

        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn6 = nn.BatchNorm3d(64)
        self.res6 = ResidualBlock3D(64)

        self.conv7 = nn.Conv3d(64, 64, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn7 = nn.BatchNorm3d(64)
        self.res7 = ResidualBlock3D(64)

        self.relu = nn.PReLU()

    def forward(self, x):
        print("Encoder")
        # input shape: (batch_size, 1, 7, 3, 10240)
        print("input shape" + str(x.shape))

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.res2(out)
        # out shape: (batch_size, 16, 7, 3, 5120)
        print("conv2 shape" + str(out.shape))

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.res3(out)
        # out shape: (batch_size, 32, 7, 3, 2560)
        print("conv3 shape" + str(out.shape))

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.res4(out)
        # out shape: (batch_size, 64, 7, 3, 1280)
        print("conv4 shape" + str(out.shape))

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.res5(out)
        # out shape: (batch_size, 64, 7, 3, 640)
        print("conv5 shape" + str(out.shape))

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.res6(out)
        # out shape: (batch_size, 64, 7, 3, 320)
        print("conv6 shape" + str(out.shape))

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)
        out = self.res7(out)
        # out shape: (batch_size, 64, 7, 3, 160)
        print("conv7 shape" + str(out.shape))

        return out


class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()

        self.transConv1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(2, 2, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.transConv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.transConv3 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn3 = nn.BatchNorm3d(32)
        self.transConv4 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn4 = nn.BatchNorm3d(16)
        self.transConv5 = nn.ConvTranspose3d(16, 16, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn5 = nn.BatchNorm3d(16)
        self.transConv6 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=(1, 1, 2), padding=1, output_padding=(0, 0, 1))

        self.relu = nn.PReLU()

    def forward(self, x):
        # input shape: (batch_size, 64, 7, 3, 160)
        print("Decoder")
        print("input shape" + str(x.shape))

        out = self.transConv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out: (batch_size, 64, 13, 5, 320)
        print("transConv1 shape" + str(out.shape))

        out = self.transConv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out shape: (batch_size, 32, 13, 5, 640)
        print("transConv2 shape" + str(out.shape))

        out = self.transConv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # out shape: (batch_size, 32, 13, 5, 1280)
        print("transConv3 shape" + str(out.shape))

        out = self.transConv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        # out shape: (batch_size, 16, 13, 5, 2560)
        print("transConv4 shape" + str(out.shape))

        out = self.transConv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        # out shape: (batch_size, 8, 13, 5, 5120)
        print("transConv5 shape" + str(out.shape))

        out = self.transConv6(out)
        # out shape: (batch_size, 1, 13, 5, 10240)
        print("transConv6 shape" + str(out.shape))

        return out


class LSTMRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, seq_len, batch_size, num_layers, dropout, learning_rate, criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        # Initialize the encoder and decoder
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout, batch_first=True)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # out shape (batch_size, 64, 7, 3, 160)
        x_shape = x.shape

        x = torch.reshape(x, (self.batch_size, x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))
        x = x.transpose(1,2) # Flatten and prepare dimensions for LSTM
        # out shape: (batch_size, 160, 1344)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Decode
        lstm_out = lstm_out.transpose(1,2)
        lstm_out = torch.reshape(lstm_out,x_shape)# Reshape for decoder

        x = self.decoder(lstm_out)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        print("Training step")
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        # Log the first validation batch for visualization
        if batch_idx == 0:
            self.validation_plots(x, y, y_hat, 'Validation')

        return loss

    def test_step(self, batch, batch_idx):
        print("Test step")
        x, y, y_path = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        # Log the first test batch for visualization
        if batch_idx == 0:
            self.validation_plots(x, y, y_hat, 'Test')

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
        # val_lr_destandardized = np.asarray(x.to('cpu'))
        # TODO: Destandardize
        val_hr_destandardized = np.asarray(y.to('cpu'))
        val_sr_destandardized = np.asarray(y_hat.to('cpu'))

        # Extract specific channels
        # lr_channel = val_lr_destandardized[0][6][2]
        val_channel = val_hr_destandardized[0][0][6][2]
        sr_channel = val_sr_destandardized[0][0][6][2]

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
        # plotly_fig.add_trace(
        # go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][2]', opacity=0.5))
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
        # lr_Y = np.fft.fft(lr_channel) / n
        # lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

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
        # freq_spec1.add_trace(
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
        val_channel = val_hr_destandardized[0][0][6][1]
        sr_channel = val_sr_destandardized[0][0][6][1]
        # lr_channel = val_lr_destandardized[0][2][6][1]

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
        # plotly_fig.add_trace(
        # go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][1]', opacity=0.5))
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
        # lr_Y = np.fft.fft(lr_channel) / n
        # lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

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
        # freq_spec1.add_trace(
        #    go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution channel', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution channel', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Predicted channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({phase + " - Frequency Spectrum - Predicted channel": freq_spec1})
