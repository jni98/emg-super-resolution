import os

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import plotly.graph_objects as go
import wandb
import numpy as np


class GAN(pl.LightningModule):
    def __init__(self, generator, discriminator, learning_rate_g, learning_rate_d, adv_weight, mean, std):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lr_gen = learning_rate_g
        self.lr_disc = learning_rate_d
        self.automatic_optimization = False  # Enable manual optimization
        self.adv_weight = adv_weight
        self.mean = mean
        self.std = std

        self.outputs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def mse_loss(self, fake, real):
        return F.mse_loss(fake, real)

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # train generator
        # generate sr data
        self.toggle_optimizer(optimizer_g)
        sr = self(x)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(y.size(0), 1)
        valid = valid.type_as(y)

        # adversarial loss is binary cross-entropy
        g_loss = self.adv_weight * self.adversarial_loss(self.discriminator(sr), valid) + self.mse_loss(sr, y)
        self.log("g_loss", g_loss, on_epoch=True, on_step=False)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(y.size(0), 1)
        valid = valid.type_as(y)

        real_loss = self.adversarial_loss(self.discriminator(y), valid)

        # how well can it label as fake?
        fake = torch.zeros(y.size(0), 1)
        fake = fake.type_as(y)

        fake_loss = self.adversarial_loss(self.discriminator(self(x).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss,  on_epoch=True, on_step=False)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        x, y, _, interpolated = batch
        sr = self.generator(x)

        real_pred = self.discriminator(y)
        fake_pred = self.discriminator(sr)

        d_loss_real = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        d_loss_fake = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (d_loss_real + d_loss_fake) / 2
        mse = self.mse_loss(sr, y)
        d_total_loss = self.adv_weight * d_loss + mse

        self.log('val_adv_loss', d_loss,  on_epoch=True, on_step=False)

        self.log('val_mse_loss', mse,  on_epoch=True, on_step=False)

        self.log('val_loss', d_total_loss,  on_epoch=True, on_step=False)

        self.outputs = {'interpolated': interpolated, 'y': y, 'sr': sr}

    def on_validation_epoch_end(self):
        # Check if it's the 10th epoch
        if self.current_epoch % 10 == 0:
            # Assuming 'outputs' contains the data necessary for plotting
            interpolated, y, sr = self.outputs['interpolated'], self.outputs['y'], self.outputs['sr']
            interpolated = self.destandardize(interpolated)
            y = self.destandardize(y)
            sr = self.destandardize(sr)
            self.validation_plots(interpolated, y, sr, 'Validation')

    def test_step(self, batch, batch_idx):
        x, y, y_path, interpolated = batch
        sr = self.generator(x)

        real_pred = self.discriminator(y)
        fake_pred = self.discriminator(sr)

        d_loss_real = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        d_loss_fake = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (d_loss_real + d_loss_fake) / 2
        mse = self.mse_loss(sr, y)
        d_total_loss = self.adv_weight * d_loss + mse

        self.log('test_mse_loss', mse,  on_epoch=True, on_step=False)
        self.log('test_loss', d_total_loss,  on_epoch=True, on_step=False)

        # Log the first test batch for visualization
        if batch_idx == 0:
            interpolated = self.destandardize(interpolated)
            y = self.destandardize(y)
            sr = self.destandardize(sr)
            self.validation_plots(interpolated, y, sr, 'Test')

        # Save the prediction for each batch individually
        results_dir = self.trainer.logger.save_dir  # or any other directory you have defined
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        y_path = y_path[0].replace("hr", "sr")
        # File name with batch index
        file_path = os.path.join(results_dir, y_path)
        sr = self.destandardize(sr)
        np.save(file_path, sr.cpu().numpy())  # Save as NumPy array

        print(f"Saved batch {batch_idx} predictions to {file_path}")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)
        return [opt_g, opt_d], []

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

    def destandardize(self, prediction):
        return prediction * self.std + self.mean

