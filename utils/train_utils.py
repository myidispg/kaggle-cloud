import numpy as np
import pandas as pd
import os
import albumentations

import torch

from .cloud_dataset import CloudDataset
from .constants import DATA_DIR, MODEL_DIR, VALIDATION_SPLIT, IMAGE_SIZE_SMALL
from .helpers import BCEDiceLoss, DiceCoefficient, DiceLoss

class Train:

    def __init__(self, model, shuffle: bool=True, print_every=1000,
     use_tensorboard: bool = True, resume: bool=False, device=torch.device('cpu')):
        """
        Train the model with the specified train and validation loaders.
        Args:
            model: The model to be used for training
            shuffle: Whether to shuffle the train and val dataloaders. Default=True
            print_every: Number of batches to process before printing the stats Default=1000
            use_tensorboard: Use Tensorboard to monitor the training. Default=True
            resume: Resume training from the latest loaded model? Default=False
            device: The PyTorch device on which to train. Default=CPU
        """
        self.model = model.to(device)
        self.shuffle = shuffle
        self.print_every = print_every
        self.use_tensorboard = use_tensorboard
        self.resume = resume
        self.device = device

        # Keep a track of all the losses and accuracies
        self.train_loss = list()
        self.class_loss = list()
        self.mask_loss = list()
        self.val_class_loss = list()
        self.val_mask_loss = list()
        self.val_class_acc = list()
        self.val_mask_acc = list()

        # Create the dataloaders
        transforms = albumentations.Compose([
            albumentations.VerticalFlip(p=0.2),
            albumentations.ElasticTransform(p=0.2),
            albumentations.GridDistortion(p=0.2),
            albumentations.HorizontalFlip(p=0.2),
            albumentations.ShiftScaleRotate(p=0.2),
            albumentations.Normalize(p=0.2)
        ])

        # Read the dataframe and instantiate the dataset
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        cloud_dataset = CloudDataset(df_train, transforms, output_img_shape=IMAGE_SIZE_SMALL)

        # Creating indices for train and validation set
        dataset_size = len(cloud_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(VALIDATION_SPLIT * dataset_size))

        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[:split], indices[split:]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size=1, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size=1, sampler=val_sampler)

        # Create the criterions and the optimziers
        self.criterion_mask = BCEDiceLoss()
        self.criterion_class = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        self.dice_coefficient = DiceCoefficient() # Used for validation

        #TODO work on tensorboard section
        #TODO work on logging the stats.

    def validate(self):
        """
         Validate the model. Return the validation scores like:
        class loss, mask loss, class accuracy, mask accuracy.
        Class accuracy is simple measured by (correct/total)*100.
        Mask accuracy is measured by the dice coefficient.
        """
        self.model.eval()

        running_mask_loss = 0
        running_class_loss = 0
        running_mask_accuracy = 0
        running_class_accuracy = 0

        for image, mask, label in self.val_loader:
            image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)

            predicted_mask, predicted_class = self.model(image)

            running_mask_loss += self.criterion_mask(predicted_mask.squeeze(), mask.squeeze())
            running_class_loss += self.criterion_class(predicted_class, label)

            _, indices = torch.max(predicted_class, dim=1)

            running_class_accuracy += (indices == label).sum().item()
            running_mask_accuracy += self.dice_coefficient(predicted_mask, mask)
        
        self.model.train()

        return running_mask_loss/ len(self.val_loader), running_mask_accuracy/ len(self.val_loader), running_class_loss/ len(self.val_loader), running_class_accuracy/ len(self.val_loader)

    def train(self, n_epochs):
        """
        Train the model. The model will be trained and the metrics will be saved in the 'training_logs' directory.
        Args:
            n_epochs: The number of epochs for which to train.
        """
        if self.resume:
            print(f'Loading the previously trained model and metrics...')
            checkpoint = torch.load(os.path.join(os.getcwd(), MODEL_DIR))
            #TODO read the previous data.

        print('Starting training...' if not self.resume else 'Resuming training...')

        for epoch in range(n_epochs):
            self.model.train()
            print(f'Epoch: {epoch}')
            running_loss = 0
            class_running_loss = 0
            mask_running_loss = 0
            for i, (image, mask, label) in enumerate(self.train_loader):
                image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)
                predicted_mask, predicted_class = self.model(image)

                # Calculate the losses.
                mask_loss = self.criterion_mask(predicted_mask.squeeze(), mask.squeeze())
                class_loss = self.criterion_class(predicted_class, label)

                total_loss = (mask_loss + class_loss) / 2 # Average the losses.

                # Append to the lists to log them.
                running_loss += total_loss.item()
                class_running_loss += class_loss.item()
                mask_running_loss += mask_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # At every "self.print_every" step, display the log TODO save the logs too.
                if (i+1) % self.print_every == 0:
                    val_mask_loss, val_mask_acc, val_class_loss, val_class_acc = self.validate()
                    print(f'Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss/(self.print_every)} ' \
                        f'Class loss: {class_running_loss/(self.print_every)}, Mask loss: {mask_running_loss/(self.print_every)}')
                    print(f'Validation Stats:\nMask loss: {val_mask_loss}, Mask accuracy: {val_mask_acc}, ' \
                            f'Class loss: {val_class_loss}, Class_accuracy: {val_class_acc}\n')
                    
                    self.train_loss.append(running_loss/self.print_every)
                    self.class_loss.append(class_running_loss/self.print_every)
                    self.mask_loss.append(mask_running_loss/self.print_every)
                    self.val_class_loss.append(val_class_loss)
                    self.val_mask_loss.append(val_mask_loss)
                    self.val_class_acc.append(val_class_acc)
                    self.val_mask_acc.append(val_mask_acc)

                    running_loss = 0
                    mask_running_loss = 0
                    class_running_loss = 0



