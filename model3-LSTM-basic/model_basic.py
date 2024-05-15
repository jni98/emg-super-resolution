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

    def __init__(self, n_features, hidden_size, seq_len, batch_size, num_layers, dropout, learning_rate, criterion,
                 mean, std):
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

        self.conv1 = nn.Conv1d(in_channels = self.n_features, out_channels = 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.PReLU()

        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        # Transpose convolution layer to upscale to 65 channels
        # Adjust kernel size and stride to match your specific scaling needs
        self.conv3 = nn.Conv1d(in_channels=self.hidden_size, out_channels=65, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        print("input shape:" , x.shape)
        out = torch.reshape(x, (self.batch_size, x.shape[1]*x.shape[2], x.shape[-1]))

        #input shape: [batch, channels, seq_len]
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))

        # Reshape for LSTM to: [batch, seq_len, channels]
        out = out.transpose(1, 2)
        lstm_out, _ = self.lstm(out)

        # Reshape for final result to [batch, channels, seq_len]
        lstm_out = lstm_out.transpose(1, 2)

        out = self.conv3(lstm_out)

        out = torch.reshape(out,(self.batch_size,13, 5, x.shape[-1] ))

        print("output shape:" , out.shape)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda _: 0.5 ** (self.current_epoch // 200)),
            'name': 'lr_scheduler_logging',
            'interval': 'epoch',  # Ensure scheduler updates at every epoch interval
            'frequency': 1  # Update the scheduler at every epoch
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        print("Training step")
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
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
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        # Return additional data for plotting
        self.outputs = {'x': x, 'y': y, 'interpolated': interpolated, 'y_hat': y_hat, 'loss': loss}

        return loss

    def test_step(self, batch, batch_idx):
        print("Test step")
        x, y, y_path, interpolated = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, sync_dist=True)

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
        interpolated_channel = interpolated_destandardized[0][6][2]
        hr_channel = hr_destandardized[0][6][2]
        sr_channel = sr_destandardized[0][6][2]

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
        hr_channel = hr_destandardized[0][6][1]
        sr_channel = sr_destandardized[0][6][1]
        interpolated_channel = interpolated_destandardized[0][6][1]

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
