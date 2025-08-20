import os
import sys
import warnings
import logging
import time
from datetime import date
import torchvision
from torchvision.transforms import transforms
import torch.optim
import torch.nn as nn
from cnn.model import CNN
sys.path.append('../')
from utils.gen_utils import create_dir, get_accuracy

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
epochs = 250
batch_size = 16
lr = 0.01
workers = 4

def main():
    """
    Main function to train the CNN model for human voice detection.

    This function:
    - Sets up the training directory and logging.
    - Defines data transformations and loads the training dataset.
    - Initializes the CNN model, loss function, and optimizer.
    - Trains the model over a specified number of epochs.
    - Logs training progress and saves the trained model.

    Returns:
        None
    """
    # create new directory for the current training session
    today = date.today()
    today = str(today.strftime('%m-%d-%Y'))
    dir_ = str(root_dir + '/saved_models/CNN/train-' + today)
    create_dir(dir_)

    log_file_name = 'CNN-' + today + '.log'
    logging.basicConfig(filename=os.path.join(dir_, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    # set basic transforms. Spectrograms have to look a certain way so rotations,
    # flips, and other transforms do not make sense in this application
    transform = {'train': transforms.Compose([transforms.Resize([32, 32]),
                                              transforms.ToTensor()])}

    # get train dataset
    train_data = torchvision.datasets.ImageFolder(root='/opt/voice_detect/data/plots/train/',
                                                  transform=transform['train'])

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)  # Pin memory for faster data transfer

    model = CNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # for each epoch
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_steps = 0

        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                prediction = model(img)
                loss = criterion(prediction, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_accuracy += get_accuracy(prediction, label)
            epoch_steps += 1

        # Average accuracy over steps
        epoch_accuracy /= epoch_steps

        # print status onto terminal and log file
        if (epoch + 1) % 10 == 0 or epoch == 0:  # Log every 10 epochs
            print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1,
                                                                    epochs,
                                                                    epoch_loss,
                                                                    epoch_accuracy))
            logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1,
                                                                           epochs,
                                                                           epoch_loss,
                                                                           epoch_accuracy))

        # save model
        model_file_name = 'CNN-' + today + '.pt'
        torch.save(model.state_dict(), os.path.join(dir_, model_file_name))

if __name__ == '__main__':
    main()
