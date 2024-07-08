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


class Trainer:
    """
    Class to train the model.
    """

    def __init__(
            self, data_directory: str, test_data_directory: str,
            results_directory: str, num_epochs: int, config, sweep_id
    ) -> None:
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
        data_directory: str
            Path to the directory where the training data is stored
        test_data_directory: str
            Path to the directory where the testing data is stored
        results_directory: str
            Path to the directory where the results should be saved
        num_epochs: int
            Number of epochs to train the model
        config: dict
            Configuration parameters for the training
        sweep_id: str
            Sweep ID for wandb
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

        self.unique_folder = '/' + datetime.now().strftime('%Y%m%d%H%M%S')
        self.betas = (0.9, 0.999)

        # Create a dictionary for the hyperparameters
        self.hyperparameter = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr
        }

        # Set the device to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # Initialize wandb
        wandb.init(project="emg_2D-super-resolution", entity="masterthesis-sr")
        print("init done")

    def prepare_data(self) -> None:
        """
        Prepares the training, validation, and testing sets.

        Returns
        -------
        None
        """
        # Load the JSON file with values for standardization
        with open(self.data_directory + '/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        # Load training and test data
        data_train_val = pd.read_csv(self.data_directory + '/data.csv', sep=";")
        data_test = pd.read_csv(self.test_data_directory + '/data.csv', sep=";")

        # Split training data into training and validation sets
        train, val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, random_state=42)

        # Create DataLoaders for training, validation, and testing sets
        self.training = DataLoader(
            Custom_Dataset(train, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=self.config.augment),
            batch_size=self.batch_size, pin_memory=True
        )

        self.validation = DataLoader(
            Custom_Dataset(val, self.data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=False),
            batch_size=self.batch_size, pin_memory=True
        )

        self.testing = DataLoader(
            Custom_Dataset(data_test, self.test_data_directory, self.mean_std_dict['mean'],
                           self.mean_std_dict['std'], augment=False), pin_memory=True)

        # Update hyperparameter dictionary
        self.hyperparameter["num_samples"] = data_train_val.shape[0]
        self.hyperparameter["standardized"] = 'True'
        self.hyperparameter["device"] = str(self.device)

        # Ensure the directory exists and save the hyperparameters
        print("Creating folder")
        os.makedirs(self.run_folder + self.unique_folder, exist_ok=True)
        with open(self.run_folder + self.unique_folder + '/hyperparameter.json', 'w') as file:
            json.dump(self.hyperparameter, file)

        print("Successfully prepared the data")

    def initialize_models(self) -> None:
        """
        Initializes the Generator, Discriminator, Optimizers, and Loss functions.

        Returns
        -------
        None
        """
        # Initialize the Generator and Discriminator
        self.generator = Generator(in_channels=1)
        self.discriminator = Discriminator()

        # If multiple GPUs are available, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        # Move models to the appropriate device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Define loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mse = nn.MSELoss()

        # Define optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.000001, betas=self.betas)

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
        val_loss = []
        psnr = []
        best_val_loss = float('inf')

        # Track gradients and parameters with wandb
        wandb.watch(self.generator, criterion='mse', log='all')
        wandb.watch(self.discriminator, criterion='mse', log='all')

        for epoch in range(1, self.num_epochs + 1):
            print("epoch", epoch)

            # Check GPU memory usage before training
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_cached()

            # Train for one epoch
            epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d = self.train_epoch(epoch)
            avg_val_loss, avg_psnr = self.validate(epoch)

            # Append losses and PSNR
            train_g_loss.append(epoch_loss_g)
            train_d_loss.append(epoch_loss_d)
            train_d_loss_real.append(epoch_loss_d_real)
            train_d_loss_fake.append(epoch_loss_d_fake)
            val_loss.append(avg_val_loss)
            psnr.append(avg_psnr)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_final_models()

            # Clear unnecessary variables to release memory
            del epoch_loss_g, epoch_loss_d_real, epoch_loss_d_fake, epoch_loss_d, avg_val_loss, avg_psnr

            # Check GPU memory usage after training
            allocated_diff = torch.cuda.memory_allocated() - initial_allocated
            cached_diff = torch.cuda.memory_cached() - initial_cached

            print(f"After epoch {epoch}, GPU memory allocated increased by: {allocated_diff} bytes")
            print(f"After epoch {epoch}, GPU memory cached increased by: {cached_diff} bytes")

        return train_g_loss, train_d_loss, train_d_loss_real, train_d_loss_fake, val_loss, psnr

    def train_epoch(self, epoch: int) -> tuple:
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

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths, lr_interpolated) in enumerate(self.training):
            print("batch index", batch_idx)

            # Move data to the appropriate device
            lr_emg = lr_emg.to(self.device)
            hr_emg = hr_emg.to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()

            # Generate super-resolved data (SR) from LR data and detach to prevent backprop through generator
            sr_emg = self.generator(lr_emg).detach().to(self.device)

            # Create real and fake labels
            real_labels = torch.ones(hr_emg.size(0), 1).to(self.device)
            fake_labels = torch.zeros(hr_emg.size(0), 1).to(self.device)

            # Calculate discriminator losses on real data
            output_discriminator_real = self.discriminator(hr_emg).to(self.device)
            loss_discriminator_real = self.adversarial_loss(output_discriminator_real, real_labels)

            # Calculate discriminator losses on fake data
            output_discriminator_fake = self.discriminator(sr_emg)
            loss_discriminator_fake = self.adversarial_loss(output_discriminator_fake, fake_labels)

            # Total discriminator loss
            loss_discriminator = loss_discriminator_real + loss_discriminator_fake
            loss_discriminator.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()

            # Generate SR data from LR data
            sr_emg = self.generator(lr_emg).to(self.device)

            # Calculate generator losses
            output = self.discriminator(sr_emg)
            loss_generator_adv = self.adversarial_loss(output, real_labels)
            loss_generator_content = self.mse(sr_emg, hr_emg)

            # Total generator loss
            loss_generator = (self.adv_loss_weight * loss_generator_adv + loss_generator_content)
            loss_generator.backward()
            self.optimizer_G.step()

            # Accumulate losses
            total_loss_g += loss_generator.item()
            total_loss_d += loss_discriminator.item()
            total_loss_d_real += loss_discriminator_real.item()
            total_loss_d_fake += loss_discriminator_fake.item()

        # Calculate average losses
        avg_loss_g = total_loss_g / len(self.training)
        avg_loss_d_real = total_loss_d_real / len(self.training)
        avg_loss_d_fake = total_loss_d_fake / len(self.training)
        avg_loss_d = total_loss_d / len(self.training)

        # Log losses to wandb
        wandb.log({'epoch': epoch, 'avg_loss_g': avg_loss_g, 'avg_loss_d_real': avg_loss_d_real,
                   'avg_loss_d_fake': avg_loss_d_fake, 'avg_loss_d': avg_loss_d})

        print(f"Epoch [{epoch}/{self.num_epochs}] Average Loss D: {avg_loss_d}, Average Loss G: {avg_loss_g}")

        return avg_loss_g, avg_loss_d_real, avg_loss_d_fake, avg_loss_d

    def validate(self, epoch: int) -> tuple:
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
        total_val_gen_content = 0.0
        total_val_gen_adv = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_lr, val_hr, val_lr_path, val_hr_path, val_lr_interpolated) in enumerate(
                    self.validation):
                print("validation batch index", val_batch_idx)

                # Move data to the appropriate device
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

                # Total loss for validation
                val_loss = (self.adv_loss_weight * loss_generator_adv +
                            loss_generator_content)

                # Accumulate validation loss
                total_validation_loss += val_loss.item()
                total_val_gen_adv += loss_generator_adv.item()
                total_val_gen_content += loss_generator_content.item()

                # Calculate PSNR
                psnr = 10 * torch.log10(1.0 / loss_generator_content)
                total_psnr += psnr.item()

                # Destandardize the data for logging
                val_hr_destandardized = destandardize_data(
                    val_hr, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )
                val_sr_destandardized = destandardize_data(
                    val_sr, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )
                val_lr_interpolated = destandardize_data(
                    val_lr_interpolated, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )

                # Log validation plots once
                if val_batch_idx == 6:
                    self.validation_plots_logging(
                        val_lr_interpolated, val_hr_destandardized,
                        val_sr_destandardized, 'Validation'
                    )

                # Calculate absolute difference for one channel
                abs_diff = torch.abs(
                    val_hr_destandardized[0][0][6][2] - val_sr_destandardized[0][0][6][2]
                )
                abs_diff = abs_diff.squeeze().to('cpu')
                abs_diff = np.asarray(abs_diff)
                total_abs_diff += abs_diff

            # Calculate average validation loss and PSNR
            avg_validation_loss = total_validation_loss / len(self.validation)
            avg_val_content = total_val_gen_content / len(self.validation)
            avg_val_adv = total_val_gen_adv / len(self.validation)
            avg_psnr = total_psnr / len(self.validation)

            # Log validation metrics to wandb
            wandb.log({'val_loss': avg_validation_loss, 'val_content': avg_val_content,
                       'val_adv': avg_val_adv, 'val_psnr': avg_psnr})

            # Log absolute difference
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

    def validation_plots_logging(
            self, val_lr_destandardized, val_hr_destandardized, val_sr_destandardized, phase
    ):
        """
        Logs validation plots using wandb.

        Parameters
        ----------
        val_lr_destandardized: torch.Tensor
            Destandardized low-resolution validation data
        val_hr_destandardized: torch.Tensor
            Destandardized high-resolution validation data
        val_sr_destandardized: torch.Tensor
            Destandardized super-resolution validation data
        phase: str
            The phase of logging (e.g., 'Validation', 'Test')

        Returns
        -------
        None
        """
        # Convert tensors to numpy arrays and squeeze the dimensions for plotting
        interpolated_destandardized = np.asarray(val_lr_destandardized.to('cpu')).squeeze()
        hr_destandardized = np.asarray(val_hr_destandardized.to('cpu')).squeeze()
        sr_destandardized = np.asarray(val_sr_destandardized.to('cpu')).squeeze()

        if phase == 'Validation':
            # Log a batch in the middle of the validation set
            interpolated_destandardized = interpolated_destandardized[self.batch_size // 2]
            hr_destandardized = hr_destandardized[self.batch_size // 2]
            sr_destandardized = sr_destandardized[self.batch_size // 2]

        # Extract specific channels
        interpolated_channel = interpolated_destandardized[6][2]
        hr_channel = hr_destandardized[6][2]
        sr_channel = sr_destandardized[6][2]

        # Log plots for the given channel
        self.log_plots(interpolated_channel, hr_channel, sr_channel, phase, 'Given Channel [6][2]')

        # Log plots for the predicted channel
        hr_channel = hr_destandardized[6][1]
        sr_channel = sr_destandardized[6][1]
        interpolated_channel = interpolated_destandardized[6][1]
        self.log_plots(interpolated_channel, hr_channel, sr_channel, phase, 'Predicted Channel [6][1]')

    def log_plots(self, interpolated_channel, hr_channel, sr_channel, phase, channel_description):
        """
        Helper function to log the plots for the given channels.

        Parameters
        ----------
        interpolated_channel: numpy array
            Interpolated channel data
        hr_channel: numpy array
            High-resolution channel data
        sr_channel: numpy array
            Super-resolution channel data
        phase: str
            The phase of logging (e.g., 'Validation', 'Test')
        channel_description: str
            Description of the channel

        Returns
        -------
        None
        """
        self.log_cross_correlation(hr_channel, sr_channel, phase, channel_description)
        self.log_time_domain_plots(interpolated_channel, hr_channel, sr_channel, phase, channel_description)
        self.log_frequency_spectrum(hr_channel, sr_channel, interpolated_channel, phase, channel_description)

    def log_cross_correlation(self, hr_channel, sr_channel, phase, channel_description):
        """
        Logs the cross-correlation plot.

        Parameters
        ----------
        hr_channel: numpy array
            High-resolution channel data
        sr_channel: numpy array
            Super-resolution channel data
        phase: str
            The phase of logging (e.g., 'Validation', 'Test')
        channel_description: str
            Description of the channel

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
            title=f'Cross Correlation - {channel_description}',
            xaxis_title='Index',
            yaxis_title='Cross Correlation',
            template="plotly_white"
        )
        wandb.log({f'{phase} Cross correlation - {channel_description}': plotly_fig})

    def log_time_domain_plots(self, interpolated_channel, hr_channel, sr_channel, phase, channel_description):
        """
        Logs the time domain plots.

        Parameters
        ----------
        interpolated_channel: numpy array
            Interpolated channel data
        hr_channel: numpy array
            High-resolution channel data
        sr_channel: numpy array
            Super-resolution channel data
        phase: str
            The phase of logging (e.g., 'Validation', 'Test')
        channel_description: str
            Description of the channel

        Returns
        -------
        None
        """
        diff_channel = hr_channel - sr_channel

        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter(x=np.arange(len(sr_channel)), y=sr_channel, name='Super-resolution'))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(hr_channel)), y=hr_channel, name='High-resolution', opacity=0.5))
        plotly_fig.add_trace(
            go.Scatter(x=np.arange(len(interpolated_channel)), y=interpolated_channel, name='Interpolated',
                       opacity=0.5))
        plotly_fig.add_trace(go.Scatter(x=np.arange(len(diff_channel)), y=diff_channel, name='HR - SR'))
        plotly_fig.update_layout(
            title=f'Super-resolution vs {phase} - {channel_description}',
            showlegend=True,
            xaxis_title="Time in samples",
            yaxis_title="Amplitude in µV"
        )
        wandb.log({f'{phase} - SR vs Val - {channel_description}': plotly_fig})

    def log_frequency_spectrum(self, hr_channel, sr_channel, interpolated_channel, phase, channel_description):
        """
        Logs the frequency spectrum plots.

        Parameters
        ----------
        hr_channel: numpy array
            High-resolution channel data
        sr_channel: numpy array
            Super-resolution channel data
        interpolated_channel: numpy array
            Interpolated channel data
        phase: str
            The phase of logging (e.g., 'Validation', 'Test')
        channel_description: str
            Description of the channel

        Returns
        -------
        None
        """
        sampling_freq = 2048

        hr_Y, sr_Y, interpolated_Y, frq = self.compute_fft(hr_channel, sr_channel, interpolated_channel, sampling_freq)

        freq_spec = go.Figure()
        freq_spec.add_trace(go.Scatter(x=frq, y=np.abs(hr_Y), mode='lines', name='High-resolution', opacity=0.5))
        freq_spec.add_trace(go.Scatter(x=frq, y=np.abs(interpolated_Y), mode='lines', name='Interpolated', opacity=0.5))
        freq_spec.add_trace(go.Scatter(x=frq, y=np.abs(sr_Y), mode='lines', name='Super-resolution', opacity=1))
        freq_spec.update_layout(
            title=f'Frequency Spectrum - {channel_description}',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude'
        )
        wandb.log({f'{phase} - Frequency Spectrum - Scatter - {channel_description}': freq_spec})

    def compute_fft(self, hr_channel, sr_channel, interpolated_channel, sampling_freq):
        """
        Computes the FFT for the given channels.

        Parameters
        ----------
        hr_channel: numpy array
            High-resolution channel data
        sr_channel: numpy array
            Super-resolution channel data
        interpolated_channel: numpy array
            Interpolated channel data
        sampling_freq: int
            Sampling frequency

        Returns
        -------
        tuple of numpy arrays
            FFT of high-resolution, super-resolution, and interpolated channels, and frequency range
        """
        n = len(hr_channel)
        k = np.arange(n)
        T = n / sampling_freq
        frq = k / T  # Two sides frequency range
        frq = frq[range(n // 2)]  # One side frequency range
        hr_Y = np.fft.fft(hr_channel) / n
        hr_Y = hr_Y[range(n // 2)]
        sr_Y = np.fft.fft(sr_channel) / n
        sr_Y = sr_Y[range(n // 2)]
        interpolated_Y = np.fft.fft(interpolated_channel) / n
        interpolated_Y = interpolated_Y[range(n // 2)]

        return hr_Y, sr_Y, interpolated_Y, frq

    def save_final_models(self) -> None:
        """
        Saves the final Generator and Discriminator.

        Returns
        -------
        None
        """
        torch.save(
            self.generator.state_dict(),
            self.run_folder + self.unique_folder + '/final_generator.pt'
        )
        torch.save(
            self.discriminator.state_dict(),
            self.run_folder + self.unique_folder + '/final_discriminator.pt'
        )

    def plot_result_losses(
            self, train_g_loss: list, train_d_loss_real: list, train_d_loss_fake: list,
            train_d_loss: list, val_loss: list, psnr: list
    ) -> None:
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
        val_loss_np = np.array(val_loss)
        psnr_np = np.array(psnr)

        # Save NumPy arrays to files
        np.save(self.run_folder + self.unique_folder + '/train_g_loss.npy', train_g_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss.npy', train_d_loss_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_real.npy', train_d_loss_real_np)
        np.save(self.run_folder + self.unique_folder + '/train_d_loss_fake.npy', train_d_loss_fake_np)
        np.save(self.run_folder + self.unique_folder + '/val_loss.npy', val_loss_np)
        np.save(self.run_folder + self.unique_folder + '/psnr.npy', psnr_np)

        # Plot Generator and Discriminator loss
        fig_generator_discriminator, (
            ax_generator_loss, ax_discriminator_loss, ax_discriminator_loss_r_f
        ) = plt.subplots(nrows=3, ncols=1)

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
        None
        """
        # Free memory due to cuda problems
        del self.discriminator
        del self.optimizer_G
        del self.optimizer_D
        del self.training
        del self.validation
        torch.cuda.empty_cache()

        print("Starting to generate SR data")

        # Load the final generator
        self.generator.load_state_dict(torch.load(self.run_folder + self.unique_folder + '/final_generator.pt'))
        self.generator.eval()  # Set the generator to evaluation mode
        total_test_loss = 0.0

        # Generate data from LR input
        with torch.no_grad():
            for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths, lr_emg_interpolated) in enumerate(self.testing):
                lr_emg = lr_emg.to(self.device)
                hr_emg = hr_emg.to(self.device)

                # Generate super-resolved data (SR) from LR data
                lr_sr_emg = self.generator(lr_emg)

                # Calculate content loss
                test_content = self.mse(lr_sr_emg, hr_emg)

                # Accumulate validation loss
                total_test_loss += test_content.item()

                # Destandardize the data for saving
                lr_sr_emg_destandardized = destandardize_data(
                    lr_sr_emg, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )
                interpolated_destandardized = destandardize_data(
                    lr_emg_interpolated, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )
                hr_emg_destandardized = destandardize_data(
                    hr_emg, self.mean_std_dict['mean'], self.mean_std_dict['std']
                )

                # Log test plots
                if batch_idx == 0:
                    self.validation_plots_logging(
                        interpolated_destandardized, hr_emg_destandardized,
                        lr_sr_emg_destandardized, 'Test'
                    )

                # Save the SR data
                lr_sr_emg_destandardized = lr_sr_emg_destandardized.to('cpu').numpy().squeeze()
                model_path = self.run_folder + self.unique_folder + '/results/'
                os.makedirs(model_path, exist_ok=True)
                np.save(model_path + lr_emg_paths[0], lr_sr_emg_destandardized)

        print("Successfully generated LR to SR data")

        # Calculate average test loss
        avg_test_content = total_test_loss / len(self.testing)
        print("Logging test loss")
        wandb.log({'test loss': avg_test_content})

        torch.cuda.empty_cache()