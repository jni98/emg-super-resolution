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
import ptwt
import wandb
import torch
import pandas as pd

from generator import Generator
from discriminator import Discriminator
from data import Custom_Dataset

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from torch.optim.lr_scheduler import LambdaLR  # Import LambdaLR



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

        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.adv_loss_weight = config.adv_loss_weight
        self.fourier_loss_weight = config.fourier_loss_weight

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
        self.lr_lambda = lambda epoch: 0.5 ** (epoch // 300)

        print("init done")

    def prepare_data(self) -> None:
        """
        Prepares the training, validation and testing set.

        Returns
        -------
        None
        """

        # Load the JSON file with values for standardization into a Python dictionary
        with open(self.data_directory + '/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        data_train_val = pd.read_csv(self.data_directory + '/data.csv', sep=";")
        data_test = pd.read_csv(self.test_data_directory + '/data.csv', sep=";")

        train, val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, random_state=42)

        # Use Custom_Dataset with standardization
        self.training = torch.utils.data.DataLoader(
            Custom_Dataset(train, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=self.config.augment),
            batch_size=self.batch_size, pin_memory=True
        )

        self.validation = torch.utils.data.DataLoader(
            Custom_Dataset(val, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=False),
            batch_size=self.batch_size, pin_memory=True
        )
        self.testing = torch.utils.data.DataLoader(
            Custom_Dataset(data_test, self.test_data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=False),
            pin_memory=True)

        self.hyperparameter["num_samples"] = data_train_val.shape[0]
        self.hyperparameter["standardized"] = 'True'
        self.hyperparameter["device"] = str(self.device)

        # Ensure the directory exists
        print("Creating folder")
        print(self.run_folder + self.unique_folder)
        os.makedirs(self.run_folder + self.unique_folder, exist_ok=True)

        # Save the dictionary to a file
        with open(self.run_folder + self.unique_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

        print("Successfully prepared the data")

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Discriminator, Optimizers and Loss functions.

        Returns
        -------
        None
        """
        self.generator = Generator(self.config.num_res_blocks, self.config.num_res_blocks, wavelet='db7',
                                   in_channels=1)
        self.discriminator = Discriminator()

        # In case multiple GPUs are available, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        # Ensure that your model and data are moved to the appropriate device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # print("Generator summary")
        # print(str(summary(self.generator, (1, 7,3,2048))))

        # print("Discriminator summary")
        # print(str(summary(self.discriminator, (1, 13, 5, 2048))))

        self.adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.mse = nn.MSELoss()  # Mean Squared error Loss

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        # Initialize the LR schedulers
        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
        self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=self.lr_lambda)

        print("Successfully initialized the models")

    def train(self) -> tuple:
        """
        Trains the complete model for a number of epochs.

        Returns
        -------
        train_g_loss: list
                    Generator loss over the epochs
        train_d_loss: list
                    Discriminator loss over the epochs
        val_loss: list
                    Validation loss over the epochs
        psnr: list
                    PSNR over the epochs
        """
        train_d_loss = []
        train_d_loss_real = []
        train_d_loss_fake = []
        train_g_loss = []
        train_fourier_loss = []
        val_loss = []
        psnr = []
        best_val_loss = float('inf')

        # Add wandb.watch() here to start tracking gradients and parameters
        wandb.watch(self.generator, log='all')
        wandb.watch(self.discriminator, log='all')

        for epoch in range(1, self.num_epochs + 1):
            print("epoch", epoch)

            # Check GPU memory usage before training
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_cached()

            epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, epoch_fourier_loss = self._train_epoch(
                epoch)
            avg_val_loss, avg_psnr = self._validate(epoch)

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
                self._save_final_models()  # Save the final models when a new best model is found

            # Clear unnecessary variables to release memory
            del epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, avg_val_loss, avg_psnr

            # Step the LR schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Assuming self.scheduler_G.get_last_lr() returns a list of learning rates
            last_lr_list = self.scheduler_G.get_last_lr()
            last_lr = last_lr_list[-1]  # Get the last learning rate
            wandb.log({'lr_scheduler': last_lr})

            # Check GPU memory usage after training
            allocated_diff = torch.cuda.memory_allocated() - initial_allocated
            cached_diff = torch.cuda.memory_cached() - initial_cached

            print(f"After epoch {epoch}, GPU memory allocated increased by: {allocated_diff} bytes")
            print(f"After epoch {epoch}, GPU memory cached increased by: {cached_diff} bytes")

        return train_g_loss, train_d_loss, train_d_loss_real, train_d_loss_fake, train_fourier_loss, val_loss, psnr

    def _train_epoch(self, epoch: int) -> tuple:
        """
        Trains the model for one epoch.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        avg_loss_g: float
                    Average loss of the generator over the training set
        avg_loss_d_real: float
                    Average loss of the discriminator over the training set when receiving real data
        avg_loss_d_fake: float
                    Average loss of the discriminator over the training set when receiving fake data
        avg_loss_d: float
                    Average loss of the discriminator over the training set
        """
        self.generator.train()
        self.discriminator.train()
        total_loss_g = 0.0
        total_loss_d_real = 0.0
        total_loss_d_fake = 0.0
        total_loss_d = 0.0
        total_fourier_loss = 0.0

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.training):
            print("batch index", batch_idx)

            lr_emg = lr_emg.to(self.device)
            hr_emg = hr_emg.to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()

            # Generate super-resolved data (SR) from LR data
            # detaching sr_emg prevents gradients from flowing back through the generator network when computing
            # the loss for the discriminator. --> ensure that during the optimization step for the discriminator,
            # only the gradients with respect to the discriminator's parameters are computed and used to update those parameters.
            sr_emg = self.generator(lr_emg).detach().to(self.device)

            # Create real and fake labels to train discriminator for both cases
            real_labels = torch.ones(hr_emg.size(0), 1).to(self.device)
            fake_labels = torch.zeros(hr_emg.size(0), 1).to(self.device)

            # calculate loss for discriminator receiving real data
            output_discriminator_real = self.discriminator(hr_emg).to(self.device)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)

            # calculate loss for discriminator receiving fake data
            output_discriminator_fake = self.discriminator(sr_emg)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)

            # calculate total loss for discriminator
            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)

            loss_discriminator.backward()

            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()

            # Generate SR data from LR data
            sr_emg = self.generator(lr_emg).to(self.device)

            # discriminator output for fake data
            output = self.discriminator(sr_emg)
            loss_generator_adv = self.adversarial_loss(output, real_labels)

            # generator content loss
            loss_generator_content = self.mse(sr_emg, hr_emg)

            # compute fourier transform
            sr_emg_fourier_real, sr_emg_fourier_imag = self.compute_fourier_transform(sr_emg)
            hr_emg_fourier_real, hr_emg_fourier_imag = self.compute_fourier_transform(hr_emg)

            loss_generator_fourier = self.mse(sr_emg_fourier_real, hr_emg_fourier_real) + self.mse(
                sr_emg_fourier_imag, hr_emg_fourier_imag)

            # Generator loss: adversarial loss + content loss + fourier loss
            loss_generator = (
                    self.adv_loss_weight * loss_generator_adv + loss_generator_content + self.fourier_loss_weight * loss_generator_fourier)

            print("loss_generator", loss_generator)
            print("loss_generator_adv", loss_generator_adv)
            print("loss_generator_content", loss_generator_content)

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

        wandb.log({'epoch': epoch, 'avg_loss_g': avg_loss_g, 'avg_loss_d_real': avg_loss_d_real,
                   'avg_loss_d_fake': avg_loss_d_fake, 'avg_loss_d': avg_loss_d,
                   'avg_fourier_loss': avg_fourier_loss})

        # Print progress per epoch
        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")

        return avg_loss_g, avg_loss_d_real, avg_loss_d_fake, avg_loss_d, avg_fourier_loss

    def _validate(self, epoch: int) -> tuple:
        """
        Validates the model by calculating the loss on the validation set.

        Parameters
        ----------
        epoch: int

        Returns
        -------
        avg_validation_loss: float
                             Average validation loss over the validation set
        avg_psnr: float
                    Average PSNR over the validation set
        """
        self.generator.eval()  # Set the generator to evaluation mode
        self.discriminator.eval()  # Set the discriminator to evaluation mode if needed

        total_validation_loss = 0.0
        total_psnr = 0.0
        total_abs_diff = 0.0
        total_val_gen_adv = 0.0
        total_val_gen_content = 0.0
        total_val_gen_fourier = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path) in enumerate(self.validation):
                print("validation batch index", val_batch_idx)

                val_lr = val_lr.to(self.device)
                val_hr = val_hr.to(self.device)

                # Generate SR data from LR data using the generator
                val_sr = self.generator(val_lr).to(self.device)

                # Calculate adversarial loss
                real_labels = torch.ones(val_hr.size(0), 1).to(self.device)
                output_disc = self.discriminator(val_sr)
                loss_generator_adv = self.adversarial_loss(output_disc, real_labels)

                # Calculate content loss
                loss_generator_content = self.mse(val_sr, val_hr)

                # compute fourier transform
                sr_emg_fourier_real, sr_emg_fourier_imag = self.compute_fourier_transform(val_sr)
                hr_emg_fourier_real, hr_emg_fourier_imag = self.compute_fourier_transform(val_hr)

                # Calculate Fourier loss
                loss_generator_fourier = self.mse(sr_emg_fourier_imag, hr_emg_fourier_imag) + self.mse(
                    sr_emg_fourier_real, hr_emg_fourier_real)

                # Total loss for validation
                val_loss = (self.adv_loss_weight * loss_generator_adv +
                            loss_generator_content +
                            self.fourier_loss_weight * loss_generator_fourier)

                # Accumulate validation loss
                total_validation_loss += val_loss.item()
                total_val_gen_adv += loss_generator_adv.item()
                total_val_gen_content += loss_generator_content.item()
                total_val_gen_fourier += loss_generator_fourier.item()

                # Accumulate validation loss
                total_validation_loss += val_loss.item()

                psnr = 10 * torch.log10(1.0 / loss_generator_content)

                total_psnr += psnr.item()

                # destacndardize the data for the logged plots
                val_hr_destandardized = self.destandardize_data(val_hr)
                val_sr_destandardized = self.destandardize_data(val_sr)
                val_lr_destandardized = self.destandardize_data(val_lr)

                # log absolute difference for one channel absolute difference
                abs_diff = torch.abs(val_hr_destandardized[0][0][6][2] - val_sr_destandardized[0][0][6][2])
                abs_diff = abs_diff.squeeze().to('cpu')
                abs_diff = np.asarray(abs_diff)
                total_abs_diff += abs_diff

                if val_batch_idx == 0:
                    self.validation_plots_logging(val_lr_destandardized, val_hr_destandardized, val_sr_destandardized,
                                                  abs_diff)

            # log validation loss and psnr
            avg_validation_loss = total_validation_loss / len(self.validation)
            avg_val_content = total_val_gen_content / len(self.validation)
            avg_val_fourier = total_val_gen_fourier / len(self.validation)
            avg_val_adv = total_val_gen_adv / len(self.validation)
            avg_psnr = total_psnr / len(self.validation)

            wandb.log({'val_loss': avg_validation_loss, 'val_content': avg_val_content,
                       'val_fourier': avg_val_fourier, 'val_adv': avg_val_adv, 'val_psnr': avg_psnr})

            # log absolute difference for one channel absolute difference
            avg_abs_diff = total_abs_diff / len(self.validation)
            fig2, axs2 = plt.subplots(1, 1, figsize=(8, 10))
            axs2.plot(range(len(avg_abs_diff)), avg_abs_diff)
            axs2.set_title('Average absolute difference')
            plt.tight_layout()
            wandb.log({"Average absolute difference": [wandb.Image(fig2)]})
            wandb.log({'Mean absolute difference': np.mean(avg_abs_diff)})

            # Close the figure
            plt.close(fig2)

            print(f"Epoch [{epoch}/{self.num_epochs}] Validation Loss: {avg_validation_loss}")

        self.generator.train()  # Set the generator back to training mode
        self.discriminator.train()  # Set the discriminator back to training mode if needed

        return avg_validation_loss, avg_psnr

    def validation_plots_logging(self, val_lr_destandardized, val_hr_destandardized, val_sr_destandardized,
                                 abs_diff):
        """

        Parameters
        ----------
        val_hr_destandardized
        val_sr_destandardized
        abs_diff

        Returns
        -------

        """
        # Convert tensors to numpy arrays and move to CPU
        val_lr_destandardized = np.asarray(val_lr_destandardized.to('cpu'))
        val_hr_destandardized = np.asarray(val_hr_destandardized.to('cpu'))
        val_sr_destandardized = np.asarray(val_sr_destandardized.to('cpu'))

        # Extract specific channels
        lr_channel = val_lr_destandardized[0][0][6][2]
        val_channel = val_hr_destandardized[0][0][6][2]
        sr_channel = val_sr_destandardized[0][0][6][2]

        ############################## Log Plots given channel########################################
        print("Logging plots for batch index 0 given channel")

        # x-axis for the following logs
        bar_fig_1, axs = plt.subplots(4, 1, figsize=(8, 10))
        x_values = range(val_hr_destandardized.shape[4])

        # log absolute difference for one channel
        axs[0].plot(x_values, abs_diff)
        axs[0].set_title('Absolute difference given channel')

        # Log one channel in the middle of the grid
        axs[1].plot(x_values, val_channel)
        axs[1].set_title('Validation channel [6][2]')

        # Log one channel in the middle of the grid
        axs[2].plot(x_values, sr_channel)
        axs[2].set_title('Super-resolution channel [6][2]')

        # log cross correlation
        auto_correlation = np.correlate(val_channel, val_channel, mode='full')
        cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        axs[3].plot(range(len(cross_correlation)), cross_correlation)
        axs[3].set_title('Cross Correlation channel [6][2]')

        plt.tight_layout()
        wandb.log({"Logged plots given Channel": [wandb.Image(bar_fig_1)]})
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
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][2]', opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - given Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")
        # Log the Plotly figure to W&B
        wandb.log({"SR vs Val - given Channel": plotly_fig})

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
        lr_Y = np.fft.fft(lr_channel) / n
        lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

        # Plot sr_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution channel', opacity=1, offset=0.1))

        bar_fig_1.update_layout(title='Frequency Spectrum - Given channel',
                                xaxis_title='Frequency (Hz)',
                                yaxis_title='Magnitude',
                                barmode='group')

        # Log figure to wandb
        wandb.log({"Frequency Spectrum - Given channel": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(val_Y), mode='lines', name='Validation (HR) channel', opacity=0.5))

        # Plot lr_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution channel', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution channel', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Given channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({"Frequency Spectrum - Given channel": freq_spec1})

        ############################## Log Plots predicted channel########################################
        print("Logging plots for batch index 0 predicted channel")

        # Select channels
        val_channel = val_hr_destandardized[0][0][6][1]
        sr_channel = val_sr_destandardized[0][0][6][1]
        lr_channel = val_lr_destandardized[0][0][6][1]

        # x-axis for the following logs
        fig1, axs1 = plt.subplots(4, 1, figsize=(8, 10))
        x_values = range(val_hr_destandardized.shape[4])

        # log absolute difference for one channel
        axs1[0].plot(x_values, np.abs(
            val_channel - sr_channel).squeeze())
        axs1[0].set_title('Absolute difference predicted channel')

        # Log one channel in the middle of the grid
        axs1[1].plot(x_values, val_channel)
        axs1[1].set_title('Validation channel [6][1]')

        # Log one channel in the middle of the grid
        axs1[2].plot(x_values, sr_channel)
        axs1[2].set_title('Super-resolution channel [6][1]')

        # log cross correlation
        auto_correlation = np.correlate(val_channel, val_channel, mode='full')
        cross_correlation = (np.correlate(val_channel, sr_channel, mode='full')) / np.max(auto_correlation)
        cross_correlation = cross_correlation.squeeze()
        axs1[3].plot(range(len(cross_correlation)), cross_correlation)
        axs1[3].set_title('Cross Correlation channel [6][1]')

        plt.tight_layout()
        wandb.log({"Logged plots predicted Channel": [wandb.Image(fig1)]})

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
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(lr_channel)), y=lr_channel, name='Low-resolution channel [6][1]', opacity=0.5))
        # Update layout
        plotly_fig.update_layout(title="Super-resolution vs Validation - predicted Channel", showlegend=True,
                                 xaxis_title="Time in samples", yaxis_title="Amplitude in µV")

        # Log the Plotly figure to W&B
        wandb.log({"SR vs Val - predicted Channel": plotly_fig})

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
        lr_Y = np.fft.fft(lr_channel) / n
        lr_Y = lr_Y[range(n // 2)]

        # Plot
        bar_fig_1 = go.Figure()

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(val_Y), name='Validation (HR) channel', opacity=1, offset=0))

        # Plot val_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(lr_Y), name='Low-resolution channel', opacity=1, offset=0.2))

        # Plot sr_channel
        bar_fig_1.add_trace(go.Bar(x=frq, y=np.abs(sr_Y), name='Super-resolution channel', opacity=1, offset=0.1))

        bar_fig_1.update_layout(title='Frequency Spectrum - Predicted channel',
                                xaxis_title='Frequency (Hz)',
                                yaxis_title='Magnitude',
                                barmode='group')

        # Log figure to wandb
        wandb.log({"Frequency Spectrum - Predicted channel": bar_fig_1})

        # Plot
        freq_spec1 = go.Figure()

        # Plot val_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(val_Y), mode='lines', name='Validation (HR) channel', opacity=0.5))

        # Plot lr_channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(lr_Y), mode='lines', name='Low-resolution channel', opacity=0.5))

        # Plot sr channel
        freq_spec1.add_trace(
            go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution channel', opacity=1))

        freq_spec1.update_layout(title='Frequency Spectrum - Predicted channel',
                                 xaxis_title='Frequency (Hz)',
                                 yaxis_title='Magnitude')

        wandb.log({"Frequency Spectrum - Predicted channel": freq_spec1})

    def compute_fourier_transform(self, data):

        # Compute the Fourier transform along the time dimension
        fourier_transformed_data = torch.fft.fftn(data, dim=-1)

        # Split the real and imaginary parts of the Fourier transformed data
        fourier_transformed_data_real = fourier_transformed_data.real
        fourier_transformed_data_imag = fourier_transformed_data.imag

        return fourier_transformed_data_real, fourier_transformed_data_imag

    def _save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(self.generator.state_dict(), self.run_folder + self.unique_folder + '/final_generator.pt')
        torch.save(self.discriminator.state_dict(), self.run_folder + self.unique_folder + '/final_discriminator.pt')

    def _plot_result_losses(self, train_g_loss: list, train_d_loss_real: list, train_d_loss_fake: list,
                            train_d_loss: list, train_fourier_loss: list, val_loss: list, psnr: list) -> None:
        """
        Plots the Generator and Discriminator loss over the epochs and the validation loss and PSNR over the epochs.

        Parameters
        ----------
        train_g_loss: list
        train_d_loss_real: list
        train_d_loss_fake: list
        train_d_loss: list
        val_loss: list
        psnr: list

        Returns
        -------
        None
        """

        # Convert lists to NumPy arrays
        train_g_loss_np = np.array(train_g_loss)
        train_d_loss_np = np.array(train_d_loss)
        train_d_loss_real_np = np.array(train_d_loss_real)
        train_d_loss_fake_np = np.array(train_d_loss_fake)
        train_fourier_loss_np = np.array(train_fourier_loss)
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        # Save NumPy arrays to files
        np.save(self.run_folder + self.unique_folder + '/train_g_loss.npy', train_g_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss.npy', train_d_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_real.npy', train_d_loss_real_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_fake.npy', train_d_loss_fake_np)
        np.save(self.run_folder + self.unique_folder + '/train_fourier_loss.npy', train_fourier_loss_np)
        np.save(self.run_folder + self.unique_folder + '/val_loss.npy', val_loss_np)
        np.save(self.run_folder + self.unique_folder + '/psnr.npy', psnr_np)

        # Plot Generator and Discriminator loss
        fig_generator_discriminator, (
            ax_generator_loss, ax_discriminator_loss, ax_discriminator_loss_r_f) = plt.subplots(nrows=3, ncols=1)

        # Plot Generator loss
        ax_generator_loss.plot(np.arange(len(train_g_loss_np)), train_g_loss_np)
        ax_generator_loss.set_yscale('log')
        ax_generator_loss.set_xlabel('Epoch')
        ax_generator_loss.set_ylabel('Loss')
        ax_generator_loss.set_title('Generator Loss over Epochs')

        # Plot Discriminator loss
        ax_discriminator_loss.plot(np.arange(len(train_d_loss_np)), train_d_loss_np)
        ax_discriminator_loss.set_yscale('log')
        ax_discriminator_loss.set_xlabel('Epoch')
        ax_discriminator_loss.set_ylabel('Loss')
        ax_discriminator_loss.set_title('Discriminator Loss over Epochs')

        # Plot Discriminator loss
        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_fake_np)), train_d_loss_fake_np, label='Fake labels')
        ax_discriminator_loss_r_f.plot(np.arange(len(train_d_loss_real_np)), train_d_loss_real_np, label='Real labels')
        ax_discriminator_loss_r_f.set_yscale('log')
        ax_discriminator_loss_r_f.set_xlabel('Epoch')
        ax_discriminator_loss_r_f.set_ylabel('Loss')
        ax_discriminator_loss_r_f.legend()
        ax_discriminator_loss_r_f.set_title('Discriminator Loss over Epochs')

        fig_generator_discriminator.tight_layout()
        fig_generator_discriminator.savefig(self.run_folder + self.unique_folder + '/training_losses.png')

        # Plot Validation loss, PSNR in a combined figure
        fig_val_psnr, (ax_val_loss, ax_psnr) = plt.subplots(nrows=2, ncols=1)

        # Plot Validation loss
        ax_val_loss.plot(np.arange(len(val_loss_np)), val_loss_np)
        ax_val_loss.set_xlabel('Epoch')
        ax_val_loss.set_ylabel('Loss')
        ax_val_loss.set_title('Validation Loss over Epochs')

        # Plot PSNR
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

        """
        # free memory
        del self.discriminator
        del self.optimizer_G
        del self.optimizer_D
        del self.adversarial_loss
        del self.mse
        del self.training
        del self.validation
        torch.cuda.empty_cache()

        print("Starting to generate LR to SR data")
        # load the final generator
        self.generator.load_state_dict(torch.load(self.run_folder + self.unique_folder + '/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)

                # Generate super-resolved data (SR) from LR data
                lr_sr_emg = self.generator(lr_emg).to('cpu').numpy().squeeze()
                lr_sr_emg_destandardized = self.destandardize_data(lr_sr_emg)

                # Save signal from lr to sr and hr to sr
                model_path = self.run_folder + self.unique_folder + '/results/'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + lr_emg_paths[0], lr_sr_emg_destandardized)

        print("Successfully generated LR to SR data")

        torch.cuda.empty_cache()
        #
        # print("Starting to generate HR to SR data")
        # # load the final generator
        # self.generator.load_state_dict(torch.load(self.run_folder + self.unique_folder + '/final_generator.pt'))
        # self.generator.eval()  # Set the generator to evaluation mode
        #
        # with torch.no_grad():
        #     for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
        #         hr_emg = hr_emg.to(self.device)
        #
        #         # Generate super-resolved data (SR) from LR data
        #         hr_sr_emg = self.generator(hr_emg).to('cpu').numpy().squeeze()
        #         hr_sr_emg_destandardized = self.destandardize_data(hr_sr_emg)
        #
        #         # Save signal from lr to sr and hr to sr
        #         model_path = self.run_folder + self.unique_folder + '/results/'
        #         os.makedirs(model_path, exist_ok=True)
        #         np.save(model_path + hr_emg_paths[0], hr_sr_emg_destandardized)

    def destandardize_data(self, standardized_data: torch.tensor) -> torch.tensor:
        """
        De-standardizes the data with the mean and standard deviation from the training set.

        Parameters
        ----------
        standardized_data: numpy array

        Returns
        -------
        destandardized_data: numpy array
        """

        return standardized_data * self.mean_std_dict['std'] + self.mean_std_dict['mean']
