from trainer import Trainer

if __name__ == "__main__":
    # TODO: try learning_rate= 0.0001
    # Create a object of the trainer
    trainer = Trainer(num_epochs=20, batch_size=8, learning_rate=0.0002)

    # Prepare the data of the trainer (read csv, split train and validation set)
    trainer.prepare_data()

    # Initialize the generator and discriminator
    trainer.initialize_models()

    # Run the training
    train_G_loss, train_D_loss, val_loss, psnr = trainer.train()

    # Plot the training results
    trainer._plot_result_losses(train_G_loss, train_D_loss, val_loss, psnr)
