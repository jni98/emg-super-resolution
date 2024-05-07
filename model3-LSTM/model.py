"""
Model used with sequence length 20480 compressed to sequence length 320 ( --> c = 64)
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

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn2 = nn.BatchNorm3d(64)
        self.res2 = ResidualBlock3D(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn3 = nn.BatchNorm3d(128)
        self.res3 = ResidualBlock3D(128)

        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn4 = nn.BatchNorm3d(128)
        self.res4 = ResidualBlock3D(128)

        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn5 = nn.BatchNorm3d(128)
        self.res5 = ResidualBlock3D(128)

        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn6 = nn.BatchNorm3d(128)
        self.res6 = ResidualBlock3D(128)

        self.conv7 = nn.Conv3d(128, 128, kernel_size=3, stride=(1,1,2), padding=1)  # Downsample
        self.bn7 = nn.BatchNorm3d(128)
        self.res7 = ResidualBlock3D(128)

        self.relu = nn.PReLU()

    def forward(self, x):
        print("Encoder")
        # input shape: (batch_size, 1, 7, 3, 40960)
        print("input shape" + str(x.shape))
        out = self.conv1(x)
        # out shape: (batch_size, 32, 7, 3, 40960)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #out = self.res2(out)
        # out shape: (batch_size, 64, 7, 3, 20480)
        print("conv2 shape" + str(out.shape))

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        #out = self.res3(out)
        # out shape: (batch_size, 128, 7, 3, 10240)
        print("conv3 shape" + str(out.shape))

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        #out = self.res4(out)
        # out shape: (batch_size, 128, 7, 3, 5120)
        print("conv4 shape" + str(out.shape))

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        #out = self.res5(out)
        # out shape: (batch_size, 128, 7, 3, 2560)
        print("conv5 shape" + str(out.shape))

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        #out = self.res6(out)
        # out shape: (batch_size, 128, 7, 3, 1280)
        print("conv6 shape" + str(out.shape))

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)
        #out = self.res7(out)
        # out shape: (batch_size, 128, 7, 3, 640)
        print("conv7 shape" + str(out.shape))

        return out


class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()

        self.transConv1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(2, 2, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn1 = nn.BatchNorm3d(64)

        self.conv1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm3d(64)

        self.transConv2 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm3d(64)

        self.transConv3 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn3 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn33 = nn.BatchNorm3d(64)

        self.transConv4 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn4 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn44 = nn.BatchNorm3d(64)

        self.transConv5 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1, 1, 2), padding=1,
                                             output_padding=(0, 0, 1))
        self.bn5 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn55 = nn.BatchNorm3d(64)

        self.transConv6 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1, 1, 2), padding=1, output_padding=(0, 0, 1))
        self.bn6 = nn.BatchNorm3d(64)

        self.lastConv = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.PReLU()

    def forward(self, x):
        # input shape: (batch_size, 64, 7, 3, 640)
        print("Decoder")
        print("input shape" + str(x.shape))

        out = self.transConv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out: (batch_size, 64, 13, 5, 1280)
        print("transConv1 shape" + str(out.shape))

        out = self.conv1(out)
        out = self.bn11(out)
        out = self.relu(out)

        out = self.transConv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out shape: (batch_size, 64, 13, 5, 2560)
        print("transConv2 shape" + str(out.shape))

        out = self.conv2(out)
        out = self.bn22(out)
        out = self.relu(out)

        out = self.transConv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # out shape: (batch_size, 64, 13, 5, 5120)
        print("transConv3 shape" + str(out.shape))

        out = self.conv3(out)
        out = self.bn33(out)
        out = self.relu(out)

        out = self.transConv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        # out shape: (batch_size, 64, 13, 5, 10240)
        print("transConv4 shape" + str(out.shape))

        out = self.conv4(out)
        out = self.bn44(out)
        out = self.relu(out)

        out = self.transConv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        # out shape: (batch_size, 64, 13, 5, 20480)
        print("transConv5 shape" + str(out.shape))

        out = self.conv5(out)
        out = self.bn55(out)
        out = self.relu(out)

        out = self.transConv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        # out shape: (batch_size, 64, 13, 5, 40960)
        print("transConv6 shape" + str(out.shape))

        out = self.lastConv(out)
        # out shape: (batch_size, 1, 13, 5, 40960)


        return out


class LSTMRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, seq_len, batch_size, num_layers, dropout, learning_rate, criterion, mean, std):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.mean = mean
        self.std = std

        # Initialize the encoder and decoder
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout, batch_first=True)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # out shape (batch_size, 128, 7, 3, 640)
        x_shape = x.shape

        x = torch.reshape(x, (self.batch_size, x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))
        x = x.transpose(1,2) # Flatten and prepare dimensions for LSTM
        # out shape: (batch_size, 640, 2688)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Decode
        lstm_out = lstm_out.transpose(1,2)
        lstm_out = torch.reshape(lstm_out,(self.batch_size, int(self.hidden_size/21), 7,3,x_shape[-1]))# Reshape for decoder

        x = self.decoder(lstm_out)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        print("Training step")
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        # Check if it's the 10th epoch
        if self.current_epoch % 500 == 0:
            # Assuming 'outputs' contains the data necessary for plotting
            x, y, interpolated, y_hat = self.outputs['x'], self.outputs['y'], self.outputs['interpolated'], \
            self.outputs['y_hat']
            y = self.destandardize(y)
            y_hat = self.destandardize(y_hat)
            interpolated = self.destandardize(interpolated)
            self.validation_plots(interpolated, y, y_hat, 'Validation')

    def validation_step(self, batch, batch_idx):
        x, y, _, interpolated = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Log loss as usual
        self.log('val_loss', loss)
        # Return additional data for plotting
        self.outputs = {'x': x, 'y': y, 'interpolated': interpolated, 'y_hat': y_hat, 'loss': loss}

        return loss

    def test_step(self, batch, batch_idx):
        print("Test step")
        x, y, y_path, interpolated = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        # Log the first test batch for visualization
        if batch_idx == 0:
            interpolated = self.destandardize(interpolated)
            y = self.destandardize(y)
            y_hat = self.destandardize(y_hat)
            self.validation_plots(interpolated, y, y_hat, 'Test')

        # Save the prediction for each batch individually
        results_dir = self.trainer.logger.save_dir  # or any other directory you have defined
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        y_path = y_path[0].replace("hr", "sr")
        # File name with batch index
        file_path = os.path.join(results_dir, y_path)
        y_hat = self.destandardize(y_hat)
        np.save(file_path, y_hat.cpu().numpy())  # Save as NumPy array

        print(f"Saved batch {batch_idx} predictions to {file_path}")

        return loss

    def destandardize(self, prediction):
        return prediction * self.std + self.mean

    def validation_plots(self, interpolated, y, y_hat, phase):

        # Convert tensors to numpy arrays and move to CPU
        interpolated_destandardized = np.asarray(interpolated.to('cpu'))
        hr_destandardized = np.asarray(y.to('cpu'))
        sr_destandardized = np.asarray(y_hat.to('cpu'))

        # Extract specific channels
        interpolated_channel = interpolated_destandardized[0][0][6][2]
        hr_channel = hr_destandardized[0][0][6][2]
        sr_channel = sr_destandardized[0][0][6][2]

        ############################## Log Plots given channel########################################
        print("Logging plots for batch index 0 given channel")

        plotly_fig0 = go.Figure()

        # log cross correlation
        auto_correlation = np.correlate(hr_channel, hr_channel, mode='full')
        cross_correlation = (np.correlate(hr_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        plotly_fig0.add_trace(go.Scatter(y=cross_correlation))

        # Update layout
        plotly_fig0.update_layout(
            title='Cross Correlation - Predicted Channel',
            xaxis_title='Index',
            yaxis_title='Cross Correlation',
            template="plotly_white"
        )

        # Log the Plotly figure to W&B
        wandb.log({phase + " Cross correlation - Given Channel [6][2]": plotly_fig0})

        # New figure for plots of channels
        # Create Plotly figure
        plotly_fig = go.Figure()
        # Add traces for all channels
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(hr_channel)), y=hr_channel, name='High-resolution', opacity=0.5))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(interpolated_channel)), y=interpolated_channel, name='Interpolated',
                       opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - Given Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")
        # Log the Plotly figure to W&B
        wandb.log({phase + " - SR vs Val - Given Channel [6][2]": plotly_fig})

        ############# Plot frequency histogram# #################
        sampling_freq = 2048

        # Compute FFT for val_channel
        n = len(hr_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T  # Two sides frequency range
        frq = frq[range(n // 2)]  # One side frequency range
        hr_Y = np.fft.fft(hr_channel) / n
        hr_Y = hr_Y[range(n // 2)]

        # Compute FFT for sr_channel
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]

        # Compute FFT for interpolated_channel
        interpolated_Y = np.fft.fft(interpolated_channel) / n
        interpolated_Y = interpolated_Y[range(n // 2)]

        # Plot
        # bar_fig_1 = go.Figure()
        #
        # # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(hr_Y), name='High-resolution', opacity=1, offset=0))
        #
        # # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(interpolated_Y), name='Interpolated', opacity=1, offset=0.2))
        #
        # # Plot sr_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution', opacity=1, offset=0.1))
        #
        # bar_fig_1.update_layout(title='Frequency Spectrum - Given channel',
        #                         xaxis_title='Frequency (Hz)',
        #                         yaxis_title='Magnitude',
        #                         barmode='group')
        #
        # # Log figure to wandb
        # wandb.log({phase + " - Frequency Spectrum - Bar - Given channel [6][2]": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(hr_Y), mode='lines', name='High-resolution', opacity=0.5))

        # Plot interpolated_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(interpolated_Y), mode='lines', name='Interpolated', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Given channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({phase + " - Frequency Spectrum - Scatter - Given channel [6][2]": freq_spec1})

        ############################## Log Plots predicted channel########################################
        print("Logging plots for batch index 0 predicted channel")

        # Select channels
        hr_channel = hr_destandardized[0][0][6][1]
        sr_channel = sr_destandardized[0][0][6][1]
        interpolated_channel = interpolated_destandardized[0][0][6][1]

        plotly_fig1 = go.Figure()

        # log cross correlation
        auto_correlation = np.correlate(hr_channel, hr_channel, mode='full')
        cross_correlation = (np.correlate(hr_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        plotly_fig1.add_trace(go.Scatter(y=cross_correlation))

        # Update layout
        plotly_fig1.update_layout(
            title='Cross Correlation - Predicted Channel',
            xaxis_title='Index',
            yaxis_title='Cross Correlation',
            template="plotly_white"
        )

        # Log the Plotly figure to W&B
        wandb.log({phase + " Cross correlation - Predicted Channel  [6][1]": plotly_fig1})

        # New figure for plots of channels
        # Create Plotly figure
        plotly_fig = go.Figure()
        # Add traces for all channels
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(hr_channel)), y=hr_channel, name='High-resolution', opacity=0.5))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(interpolated_channel)), y=interpolated_channel, name='Interpolated',
                       opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - Predicted Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")

        # Log the Plotly figure to W&B
        wandb.log({phase + " - SR vs Val - Predicted Channel [6][1]": plotly_fig})

        ############# Plot frequency histogram# #################
        # Generate sample data for demonstration purposes
        sampling_freq = 2048

        # Compute FFT for val_channel
        n = len(hr_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T  # Two sides frequency range
        frq = frq[range(n // 2)]  # One side frequency range
        hr_Y = np.fft.fft(hr_channel) / n
        hr_Y = hr_Y[range(n // 2)]

        # Compute FFT for sr_channel
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]

        # Compute FFT for interpolated_channel
        interpolated_Y = np.fft.fft(interpolated_channel) / n
        interpolated_Y = interpolated_Y[range(n // 2)]

        # Plot
        # bar_fig_1 = go.Figure()
        #
        # # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(hr_Y), name='High-resolution', opacity=1, offset=0))
        #
        # # Plot val_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(interpolated_Y), name='Interpolated', opacity=1, offset=0.2))
        #
        # # Plot sr_channel
        # bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution', opacity=1, offset=0.1))
        #
        # bar_fig_1.update_layout(title='Frequency Spectrum - Predicted channel',
        #                         xaxis_title='Frequency (Hz)',
        #                         yaxis_title='Magnitude',
        #                         barmode='group')
        #
        # # Log figure to wandb
        # wandb.log({phase + " - Frequency Spectrum - Bar - Predicted channel [6][1]": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(hr_Y), mode='lines', name='High-resolution', opacity=0.5))

        # Plot interpolated_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(interpolated_Y), mode='lines', name='Interpolated', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Predicted channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({phase + " - Frequency Spectrum - Scatter - Predicted channel [6][1]": freq_spec1})
