from datetime import datetime

from trainer import Trainer
import os

if __name__ == "__main__":

    # Create a directory for the run if it doesn't exist
    run_folder = datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists('runs/' + run_folder):
        os.makedirs('runs/' + run_folder)
        print(f"Folder created at: runs/{run_folder}")

    # Create a object of the trainer
    trainer = Trainer(num_epochs=30, batch_size=16, learning_rate=0.0001, run_folder='runs/' + run_folder)

    # Prepare the data of the trainer (read csv, split train and validation set)
    trainer.prepare_data()

    # Initialize the generator and discriminator
    trainer.initialize_models()

    # Run the training
    train_G_loss, train_D_loss, train_D_loss_real, train_D_loss_fake, val_loss, psnr = trainer.train()

    # Plot the training results
    trainer._plot_result_losses(train_G_loss, train_D_loss_real, train_D_loss_fake, train_D_loss, val_loss, psnr)

    trainer.generate_sr_data()
