"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Trainer. It is used to train the GAN and to generate super resolution
data.

The code is called from the train.py file which is started by the main function.
"""

import json
import os
from datetime import datetime
import matplotlib
import wandb
import torch
import pandas as pd

from generator import Generator
from discriminator import Discriminator
from data import Custom_Dataset
from utils import destandardize_data

matplotlib.use('Agg')  # Use the 'Agg' backend for matplotlib
import matplotlib.pyplot as plt

import numpy as np
import plotly.graph_objects as go

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    """
    Class to train the model.
    """

    def __init__(self, data_directory: str, test_data_directory: str, results_directory: str, num_epochs: int,
                 config, sweep_id) -> None:
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
        data_directory: str
            The directory where the training data is stored
        test_data_directory: str
            The directory where the testing data is stored
        results_directory: str
            The directory where the results are stored
        num_epochs: int
            The number of epochs to train the model
        config: Config
            The configuration object to run weights and biases sweeps
        sweep_id: str
            The sweep id to run weights and biases sweeps
        """
        self.data_directory = data_directory
        self.test_data_directory = test_data_directory
        self.run_folder = results_directory
        self.num_epochs = num_epochs
        self.config = config
        self.sweep_id = sweep_id

        # Initialize learning rate, batch size, and loss weights from config
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.adv_loss_weight = config.adv_loss_weight
        self.fourier_loss_weight = config.fourier_loss_weight
        self.bandpass_upper = config.bandpass_upper

        # Create a unique folder name based on the current date and time
        self.unique_folder = '/' + datetime.now().strftime('%Y%m%d%H%M%S')

        self.betas = (0.9, 0.999)

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        wandb.init(project="GAN-Training", entity="masterthesis-sr")

        # Define lambda function for the LR scheduler
        self.lr_lambda = lambda epoch: 0.5 ** (epoch // 100)

        print("Initialization done")

    def prepare_data(self) -> None:
        """
        Prepares the training, validation, and testing set.

        Returns
        -------
        None
        """
        # Load the JSON file with values for standardization into a Python dictionary
        with open(os.path.join(self.data_directory, 'mean_std.json'), 'r') as file:
            self.mean_std_dict = json.load(file)

        # Load training/validation and test data from CSV files
        data_train_val = pd.read_csv(os.path.join(self.data_directory, 'data.csv'), sep=";")
        data_test = pd.read_csv(os.path.join(self.test_data_directory, 'data.csv'), sep=";")

        # Split the data into training and validation sets
        train, val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, random_state=42)

        # Create DataLoaders for training, validation, and testing sets
        self.training = DataLoader(
            Custom_Dataset(train, self.data_directory, self.mean_std_dict['mean'], self.mean_std_dict['std'],
                           augment=self.config.augment, bandpass_upper=self.bandpass_upper),
            batch_size=self.batch_size, pin_memory=True
        )

        self.validation = DataLoader(
            Custom_Dataset(val, self.data_directory, self.mean_std_dict['mean'], self.mean_std_dict['std'],
                           augment=False, bandpass_upper=self.bandpass_upper),
            batch_size=self.batch_size, pin_memory=True
        )

        self.testing = DataLoader(
            Custom_Dataset(data_test, self.test_data_directory, self.mean_std_dict['mean'], self.mean_std_dict['std'],
                           augment=False, bandpass_upper=self.bandpass_upper),
            pin_memory=True
        )

        # Update hyperparameters with the number of samples and device information
        self.hyperparameter["num_samples"] = data_train_val.shape[0]
        self.hyperparameter["standardized"] = 'True'
        self.hyperparameter["device"] = str(self.device)

        # Ensure the directory for results exists
        os.makedirs(self.run_folder + self.unique_folder, exist_ok=True)

        # Save the hyperparameters to a JSON file
        with open(os.path.join(self.run_folder + self.unique_folder, 'hyperparameter.json'), 'w') as file:
            json.dump(self.hyperparameter, file)

        print("Data preparation done")

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Discriminator, Optimizers, and Loss functions.

        Returns
        -------
        None
        """
        # Initialize Generator and Discriminator models
        self.generator = Generator(self.config.num_res_blocks, self.config.num_res_blocks, wavelet='db7', in_channels=1)
        self.discriminator = Discriminator()

        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        # Move models to the appropriate device (GPU/CPU)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Define loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mse = nn.MSELoss()

        # Initialize optimizers for Generator and Discriminator
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        # Initialize learning rate schedulers
        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
        self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=self.lr_lambda)

        print("Model initialization done")

    def train(self) -> tuple:
        """
        Trains the complete model for a number of epochs.

        Returns
        -------
        tuple: Containing lists of generator loss, discriminator loss, real discriminator loss, fake discriminator loss,
               fourier loss, validation loss, and PSNR over the epochs.
        """
        train_d_loss, train_d_loss_real, train_d_loss_fake = [], [], []
        train_g_loss, train_fourier_loss, val_loss, psnr = [], [], [], []
        best_val_loss = float('inf')

        wandb.watch(self.generator, log='all')
        wandb.watch(self.discriminator, log='all')

        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}")

            # Check initial GPU memory usage
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_cached()

            # Train for one epoch and validate
            epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, epoch_fourier_loss = self.train_epoch(
                epoch)
            avg_val_loss, avg_psnr = self.validate(epoch)

            # Append losses and PSNR
            train_g_loss.append(epoch_loss_g)
            train_d_loss.append(epoch_loss_d)
            train_d_loss_real.append(epoch_loss_d_real)
            train_d_loss_fake.append(epoch_loss_d_fake)
            train_fourier_loss.append(epoch_fourier_loss)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_final_models()

            # Clear unnecessary variables to release memory
            del epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, avg_val_loss, avg_psnr

            # Step the learning rate schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Log current learning rate
            last_lr = self.scheduler_G.get_last_lr()[-1]
            wandb.log({'lr_scheduler': last_lr})

            # Check GPU memory usage after training
            allocated_diff = torch.cuda.memory_allocated() - initial_allocated
            cached_diff = torch.cuda.memory_cached() - initial_cached

            print(f"After epoch {epoch}, GPU memory allocated increased by: {allocated_diff} bytes")
            print(f"After epoch {epoch}, GPU memory cached increased by: {cached_diff} bytes")

        return train_g_loss, train_d_loss, train_d_loss_real, train_d_loss_fake, train_fourier_loss, val_loss, psnr

    def train_epoch(self, epoch: int) -> tuple:
        """
        Trains the model for one epoch.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        tuple: Containing average losses for generator, real and fake discriminator, and fourier loss.
        """
        self.generator.train()
        self.discriminator.train()

        total_loss_g, total_loss_d_real, total_loss_d_fake = 0.0, 0.0, 0.0
        total_loss_d, total_fourier_loss = 0.0, 0.0

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.training):
            lr_emg, hr_emg = lr_emg.to(self.device), hr_emg.to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()
            sr_emg = self.generator(lr_emg).detach()

            # Discriminator loss on real and fake data
            real_labels = torch.ones(hr_emg.size(0), 1).to(self.device)
            fake_labels = torch.zeros(hr_emg.size(0), 1).to(self.device)
            output_discriminator_real = self.discriminator(hr_emg)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)
            output_discriminator_fake = self.discriminator(sr_emg)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)
            loss_discriminator = loss_discriminator_real + loss_discriminator_fake

            # Update discriminator
            loss_discriminator.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()
            sr_emg = self.generator(lr_emg)
            output = self.discriminator(sr_emg)

            # Generator loss based on adversarial loss, mse, and Fourier
            loss_generator_adv = self.adversarial_loss(output, real_labels)
            loss_generator_content = self.mse(sr_emg, hr_emg)
            sr_emg_fourier_real, sr_emg_fourier_imag = self.compute_fourier_transform(sr_emg)
            hr_emg_fourier_real, hr_emg_fourier_imag = self.compute_fourier_transform(hr_emg)
            loss_generator_fourier = self.mse(sr_emg_fourier_real, hr_emg_fourier_real)
            loss_generator = (self.adv_loss_weight * loss_generator_adv + loss_generator_content +
                              self.fourier_loss_weight * loss_generator_fourier)

            # Update generator
            loss_generator.backward()
            self.optimizer_G.step()

            total_loss_g += loss_generator.item()
            total_loss_d += loss_discriminator.item()
            total_loss_d_real += loss_discriminator_real.item()
            total_loss_d_fake += loss_discriminator_fake.item()
            total_fourier_loss += loss_generator_fourier.item()

        avg_loss_g = total_loss_g / len(self.training)
        avg_loss_d_real = total_loss_d_real / len(self.training)
        avg_loss_d_fake = total_loss_d_fake / len(self.training)
        avg_loss_d = total_loss_d / len(self.training)
        avg_fourier_loss = total_fourier_loss / len(self.training)

        wandb.log({
            'epoch': epoch,
            'avg_loss_g': avg_loss_g,
            'avg_loss_d_real': avg_loss_d_real,
            'avg_loss_d_fake': avg_loss_d_fake,
            'avg_loss_d': avg_loss_d,
            'avg_fourier_loss': avg_fourier_loss
        })

        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")
        return avg_loss_g, avg_loss_d_real, avg_loss_d_fake, avg_loss_d, avg_fourier_loss

    def validate(self, epoch: int) -> tuple:
        """
        Validates the model by calculating the loss on the validation set.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        tuple: Containing average validation loss and PSNR over the validation set.
        """
        self.generator.eval()
        self.discriminator.eval()

        total_validation_loss, total_val_gen_adv = 0.0, 0.0
        total_val_gen_content, total_val_gen_fourier = 0.0, 0.0
        total_psnr, total_abs_diff = 0.0, 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                val_lr, val_hr = val_lr.to(self.device), val_hr.to(self.device)
                val_sr = self.generator(val_lr)

                real_labels = torch.ones(val_hr.size(0), 1).to(self.device)
                output_disc = self.discriminator(val_sr)

                # Generator loss based on adversarial, mse, and fourier loss
                loss_generator_adv = self.adversarial_loss(output_disc, real_labels)
                loss_generator_content = self.mse(val_sr, val_hr)
                sr_emg_fourier_real, sr_emg_fourier_imag = self.compute_fourier_transform(val_sr)
                hr_emg_fourier_real, hr_emg_fourier_imag = self.compute_fourier_transform(val_hr)
                loss_generator_fourier = self.mse(sr_emg_fourier_real, hr_emg_fourier_real)
                val_loss = (self.adv_loss_weight * loss_generator_adv + loss_generator_content +
                            self.fourier_loss_weight * loss_generator_fourier)

                total_validation_loss += val_loss.item()
                total_val_gen_adv += loss_generator_adv.item()
                total_val_gen_content += loss_generator_content.item()
                total_val_gen_fourier += loss_generator_fourier.item()
                psnr = 10 * torch.log10(1.0 / loss_generator_content)
                total_psnr += psnr.item()

                val_hr_destandardized = destandardize_data(val_hr, self.mean_std_dict['mean'],
                                                           self.mean_std_dict['std'])
                val_sr_destandardized = destandardize_data(val_sr, self.mean_std_dict['mean'],
                                                           self.mean_std_dict['std'])
                val_lr_destandardized = destandardize_data(val_lr, self.mean_std_dict['mean'],
                                                           self.mean_std_dict['std'])
                abs_diff = torch.abs(val_hr_destandardized[0][0][6][2] - val_sr_destandardized[0][0][6][2])
                abs_diff = abs_diff.squeeze().to('cpu').numpy()
                total_abs_diff += abs_diff

                # Log plots ones
                if val_batch_idx == 0:
                    self.validation_plots_logging(val_lr_destandardized, val_hr_destandardized, val_sr_destandardized,
                                                  'Validation')

            avg_validation_loss = total_validation_loss / len(self.validation)
            avg_val_content = total_val_gen_content / len(self.validation)
            avg_val_fourier = total_val_gen_fourier / len(self.validation)
            avg_val_adv = total_val_gen_adv / len(self.validation)
            avg_psnr = total_psnr / len(self.validation)

            wandb.log({
                'val_loss': avg_validation_loss,
                'val_content': avg_val_content,
                'val_fourier': avg_val_fourier,
                'val_adv': avg_val_adv,
                'val_psnr': avg_psnr
            })

            avg_abs_diff = total_abs_diff / len(self.validation)
            fig2, axs2 = plt.subplots(1, 1, figsize=(8, 10))
            axs2.plot(range(len(avg_abs_diff)), avg_abs_diff)
            axs2.set_title('Average absolute difference')
            plt.tight_layout()
            wandb.log({"Average absolute difference": [wandb.Image(fig2)]})
            wandb.log({'Mean absolute difference': np.mean(avg_abs_diff)})

            plt.close(fig2)

            print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")

        self.generator.train()
        self.discriminator.train()
        return avg_validation_loss, avg_psnr

    def compute_fourier_transform(data):
        """
        Computes the Fourier transform along the time dimension.

        Parameters
        ----------
        data: torch.Tensor
            The input data

        Returns
        -------
        tuple
            Real and imaginary parts of the Fourier transformed data
        """
        fourier_transformed_data = torch.fft.fftn(data, dim=-1)
        return fourier_transformed_data.real, fourier_transformed_data.imag

    def validation_plots_logging(self, val_lr_destandardized, val_hr_destandardized, val_sr_destandardized, phase):
        """
        Handles logging of validation plots.

        Parameters
        ----------
        val_lr_destandardized: torch.Tensor
            Destandardized low-resolution data
        val_hr_destandardized: torch.Tensor
            Destandardized high-resolution data
        val_sr_destandardized: torch.Tensor
            Destandardized super-resolution data
        phase: str
            The current phase (e.g., 'Validation' or 'Test')
        Returns
        -------
        None
        """
        val_lr_destandardized = np.asarray(val_lr_destandardized.to('cpu'))
        val_hr_destandardized = np.asarray(val_hr_destandardized.to('cpu'))
        val_sr_destandardized = np.asarray(val_sr_destandardized.to('cpu'))

        # Log for channel [6][2] --> given channel
        self.log_channel_metrics(val_lr_destandardized, val_hr_destandardized, val_sr_destandardized, phase, 6, 2)

        # Log for channel [6][1] --> predicted channel
        self.log_channel_metrics(val_lr_destandardized, val_hr_destandardized, val_sr_destandardized, phase, 6, 1)

    def log_channel_metrics(self, val_lr_destandardized, val_hr_destandardized, val_sr_destandardized, phase, row,
                            column):
        """
        Logs all metrics for a specific channel and position.

        Parameters
        ----------
        val_lr_destandardized: ndarray
            Destandardized low-resolution data
        val_hr_destandardized: ndarray
            Destandardized high-resolution data
        val_sr_destandardized: ndarray
            Destandardized super-resolution data
        phase: str
            The current phase (e.g., 'Validation' or 'Test')
        channel: int
            The channel index to log
        position: int
            The position index to log

        Returns
        -------
        None
        """
        hr_channel = val_hr_destandardized[0][0][row][column]
        sr_channel = val_sr_destandardized[0][0][row][column]
        lr_channel = val_lr_destandardized[0][0][row][column]

        diff_channel = hr_channel - sr_channel

        self.log_cross_correlation_plot(hr_channel, sr_channel, phase, f'Channel [{row}][{column}]')
        self.log_channel_comparison_plot(sr_channel, hr_channel, lr_channel, diff_channel, phase,
                                         f'Channel [{row}][{column}]')
        self.log_frequency_spectrum_plot(hr_channel, sr_channel, lr_channel, phase, f'Channel [{row}][{column}]')

    def log_cross_correlation_plot(self, hr_channel, sr_channel, phase, channel_label):
        """
        Logs cross-correlation plots.

        Parameters
        ----------
        hr_channel: np.ndarray
            High-resolution channel data
        sr_channel: np.ndarray
            Super-resolution channel data
        phase: str
            The current phase (e.g., 'Validation' or 'Test')
        channel_label: str
            Label for the channel being logged

        Returns
        -------
        None
        """
        auto_correlation = np.correlate(hr_channel, hr_channel, mode='full')
        cross_correlation = (np.correlate(hr_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()

        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter(y=cross_correlation))

        plotly_fig.update_layout(
            title=f'Cross Correlation - {phase} {channel_label}',
            xaxis_title='Index',
            yaxis_title='Cross Correlation',
            template="plotly_white"
        )

        wandb.log({f'{phase} Cross correlation - {channel_label}': plotly_fig})

    def log_channel_comparison_plot(self, sr_channel, hr_channel, lr_channel, diff_channel, phase, channel_label):
        """
        Logs channel comparison plots.

        Parameters
        ----------
        sr_channel: np.ndarray
            Super-resolution channel data
        hr_channel: np.ndarray
            High-resolution channel data
        lr_channel: np.ndarray
            Low-resolution channel data
        diff_channel: np.ndarray
            Difference between high-resolution and super-resolution data
        phase: str
            The current phase (e.g., 'Validation' or 'Test')
        channel_label: str
            Label for the channel being logged

        Returns
        -------
        None
        """
        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(hr_channel)), y=hr_channel, name='High-resolution', opacity=0.5))
        plotly_fig.add_trace(go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution', opacity=0.5))
        plotly_fig.add_trace(go.Scatter(x=np.arange(len(diff_channel)), y=diff_channel, name='HR - SR'))

        plotly_fig.update_layout(
            title=f'Super-resolution vs {phase} - {channel_label}',
            showlegend=True,
            xaxis_title="Time in samples",
            yaxis_title="Amplitude in µV"
        )

        wandb.log({f'{phase} - SR vs {phase} - {channel_label}': plotly_fig})

    def log_frequency_spectrum_plot(self, hr_channel, sr_channel, lr_channel, phase, channel_label):
        """
        Logs frequency spectrum plots.

        Parameters
        ----------
        hr_channel: np.ndarray
            High-resolution channel data
        sr_channel: np.ndarray
            Super-resolution channel data
        lr_channel: np.ndarray
            Low-resolution channel data
        phase: str
            The current phase (e.g., 'Validation' or 'Test')
        channel_label: str
            Label for the channel being logged

        Returns
        -------
        None
        """
        sampling_freq = 2048
        n = len(hr_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T
        frq = frq[range(n // 2)]
        hr_Y = np.fft.fft(hr_channel) / n
        hr_Y = hr_Y[range(n // 2)]
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]
        lr_Y = np.fft.fft(lr_channel) / n
        lr_Y = lr_Y[range(n // 2)]

        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter(x=frq, y=np.abs(hr_Y), mode='lines', name='High-resolution', opacity=0.5))
        plotly_fig.add_trace(go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution', opacity=0.5))
        plotly_fig.add_trace(go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution', opacity=1))

        plotly_fig.update_layout(
            title=f'Frequency Spectrum - {phase} {channel_label}',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude'
        )

        wandb.log({f'{phase} - Frequency Spectrum - {channel_label}': plotly_fig})

    def save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder + self.unique_folder + '/final_generator.pt')
        torch.save(self.discriminator.state_dict(), self.run_folder + self.unique_folder + '/final_discriminator.pt')

    def plot_result_losses(self, train_g_loss: list, train_d_loss_real: list, train_d_loss_fake: list,
                           train_d_loss: list, train_fourier_loss: list, val_loss: list, psnr: list) -> None:
        """
        Plots the Generator and Discriminator loss over the epochs and the validation loss and PSNR over the epochs.

        Parameters
        ----------
        train_g_loss: list
            List of generator losses over the epochs
        train_d_loss_real: list
            List of discriminator losses for real data over the epochs
        train_d_loss_fake: list
            List of discriminator losses for fake data over the epochs
        train_d_loss: list
            List of total discriminator losses over the epochs
        train_fourier_loss: list
            List of Fourier losses over the epochs
        val_loss: list
            List of validation losses over the epochs
        psnr: list
            List of PSNR values over the epochs

        Returns
        -------
        None
        """
        train_g_loss_np = np.array(train_g_loss)
        train_d_loss_np = np.array(train_d_loss)
        train_d_loss_real_np = np.array(train_d_loss_real)
        train_d_loss_fake_np = np.array(train_d_loss_fake)
        train_fourier_loss_np = np.array(train_fourier_loss)
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        np.save(self.run_folder + self.unique_folder + '/train_g_loss.npy', train_g_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss.npy', train_d_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_real.npy', train_d_loss_real_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_fake.npy', train_d_loss_fake_np)
        np.save(self.run_folder + self.unique_folder + '/train_fourier_loss.npy', train_fourier_loss_np)
        np.save(self.run_folder + self.unique_folder + '/val_loss.npy', val_loss_np)
        np.save(self.run_folder + self.unique_folder + '/psnr.npy', psnr_np)

        fig_generator_discriminator, (
            ax_generator_loss, ax_discriminator_loss, ax_discriminator_loss_r_f) = plt.subplots(nrows=3, ncols=1)

        ax_generator_loss.plot(np.arange(len(train_g_loss_np)), train_g_loss_np)
        ax_generator_loss.set_yscale('log')
        ax_generator_loss.set_xlabel('Epoch')
        ax_generator_loss.set_ylabel('Loss')
        ax_generator_loss.set_title('Generator Loss over Epochs')

        ax_discriminator_loss.plot(np.arange(len(train_d_loss_np)), train_d_loss_np)
        ax_discriminator_loss.set_yscale('log')
        ax_discriminator_loss.set_xlabel('Epoch')
        ax_discriminator_loss.set_ylabel('Loss')
        ax_discriminator_loss.set_title('Discriminator Loss over Epochs')

        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_fake_np)), train_d_loss_fake_np, label='Fake labels')
        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_real_np)), train_d_loss_real_np, label='Real labels')
        ax_discriminator_loss_r_f.set_yscale('log')
        ax_discriminator_loss_r_f.set_xlabel('Epoch')
        ax_discriminator_loss_r_f.set_ylabel('Loss')
        ax_discriminator_loss_r_f.legend()
        ax_discriminator_loss_r_f.set_title('Discriminator Loss over Epochs')

        fig_generator_discriminator.tight_layout()
        fig_generator_discriminator.savefig(self.run_folder + self.unique_folder + '/training_losses.png')

        fig_val_psnr, (ax_val_loss, ax_psnr) = plt.subplots(nrows=2, ncols=1)

        ax_val_loss.plot(np.arange(len(val_loss_np)), val_loss_np)
        ax_val_loss.set_xlabel('Epoch')
        ax_val_loss.set_ylabel('Loss')
        ax_val_loss.set_title('Validation Loss over Epochs')

        ax_psnr.plot(np.arange(len(psnr_np)), psnr_np)
        ax_psnr.set_xlabel('Epoch')
        ax_psnr.set_ylabel('PSNR')
        ax_psnr.set_title('PSNR over Epochs')

        fig_val_psnr.tight_layout()
        fig_val_psnr.savefig(self.run_folder + self.unique_folder + '/validation_loss_psnr.png')

    def generate_sr_data(self) -> None:
        """
        Generates super-resolved data from the test dataset and saves it to a file.

        Returns
        -------
        None
        """
        del self.discriminator
        del self.optimizer_G
        del self.optimizer_D
        del self.training
        del self.validation
        torch.cuda.empty_cache()

        print("Starting to generate SR data")
        self.generator.load_state_dict(torch.load(self.run_folder + self.unique_folder + '/final_generator.pt'))

        self.generator.eval()
        total_test_loss = 0.0

        # Generate data from LR input
        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

                lr_sr_emg = self.generator(lr_emg)
                test_content = self.mse(lr_sr_emg, hr_emg)
                total_test_loss += test_content.item()

                lr_sr_emg_destandardized = destandardize_data(lr_sr_emg, self.mean_std_dict['mean'],
                                                              self.mean_std_dict['std'])
                interpolated_destandardized = destandardize_data(lr_emg, self.mean_std_dict['mean'],
                                                                 self.mean_std_dict['std'])
                hr_emg_destandardized = destandardize_data(hr_emg, self.mean_std_dict['mean'],
                                                           self.mean_std_dict['std'])

                if batch_idx == 0:
                    self.validation_plots_logging(interpolated_destandardized, hr_emg_destandardized,
                                                  lr_sr_emg_destandardized, 'Test')

                lr_sr_emg_destandardized = lr_sr_emg_destandardized.to('cpu').numpy().squeeze()
                model_path = self.run_folder + self.unique_folder + '/results/'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + lr_emg_paths[0], lr_sr_emg_destandardized)

        print("Successfully generated LR to SR data")

        avg_test_content = total_test_loss / len(self.testing)
        print("Logging test loss")
        wandb.log({'test loss': avg_test_content})

        torch.cuda.empty_cache()