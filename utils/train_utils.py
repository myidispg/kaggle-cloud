import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import albumentations

import torch
from torch.utils.tensorboard import SummaryWriter

from .cloud_dataset import CloudDataset
from .constants import DATA_DIR, MODEL_DIR, VALIDATION_SPLIT, IMAGE_SIZE_SMALL, IDX2LABELS
from .helpers import BCEDiceLoss, DiceCoefficient, DiceLoss

class Train:

    def __init__(self, model, shuffle: bool=True, print_every=1000, log_every=100,
    batch_size = 1, use_tensorboard: bool = True, resume: bool=False,
    device=torch.device('cpu')):
        """
        Train the model with the specified train and validation loaders.
        Args:
            model: The model to be used for training
            shuffle: Whether to shuffle the train and val dataloaders. Default=True
            print_every: Number of batches to process before printing the stats Default=1000
            log_every: Number of batches after which to log stats into Tensorboard
            batch_size: The number of data points processed before weights are updated.
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
        self.batch_size = batch_size

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
            # albumentations.VerticalFlip(p=0.2),
            albumentations.ElasticTransform(p=0.2),
            albumentations.GridDistortion(p=0.2),
            albumentations.HorizontalFlip(p=0.2),
            albumentations.ShiftScaleRotate(p=0.2),
            # albumentations.Normalize(p=0.2)
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

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size=batch_size, sampler=val_sampler)

        # Create the criterions and the optimziers
        self.criterion_mask = BCEDiceLoss()
        self.criterion_class = torch.nn.CrossEntropyLoss()

        self.optimizer = RAdam(self.model.parameters(), lr=1e-2)

        # Reduce LR on Plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode='min',
                                                              factor=1e-1,
                                                              verbose=True, 
                                                              patience=1,
                                                              )

        self.dice_coefficient = DiceCoefficient() # Used for validation

        #TODO work on tensorboard section
        if self.use_tensorboard:
            # Create a tensorboard SummaryWriter()
            self.tensorboard_writer = SummaryWriter()
            # self.tensorboard_writer.add_graph(self.model)

        # Tensorboard should be usable as soon as the object is instantiated. 
        # This allows prediction on the model too before need for training. 
        if self.resume:
            self.read_saved_state()

    def save_model(self):
        """
        Save the model and stats. For resuming later.        
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'class_loss': self.class_loss,
            'mask_loss': self.mask_loss,
            'val_class_loss': self.val_class_loss,
            'val_class_acc': self.val_class_acc,
            'val_mask_loss': self.val_mask_loss,
            'val_mask_acc': self.val_mask_acc
        }

        torch.save(checkpoint, os.path.join(os.getcwd(), MODEL_DIR, 'efficientnet-b2_unet.pth'))

    def read_saved_state(self):
        """
        Read the model's state and variables into the class variables.
        """
        print('Reading the saved state...')
        checkpoint = torch.load(os.path.join(os.getcwd(), MODEL_DIR, 'unet.pth'))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkkpoint['scheduler'])
        self.train_loss = checkpoint['train_loss']
        self.class_loss = checkpoint['class_loss']
        self.mask_loss = checkpoint['mask_loss']
        self.val_class_loss = checkpoint['val_class_loss']
        self.val_class_acc = checkpoint['val_class_acc']
        self.val_mask_loss = checkpoint['val_mask_loss']
        self.val_mask_acc = checkpoint['val_mask_acc']
        print('Loaded the model state and metrics!')

        if self.use_tensorboard:
            # Read the stuff into tensorboard too.
            # for (train_loss, class_loss, mask_loss) in zip(self.train_loss, self.class_loss, self.mask_loss):
            #     self.tensorboard_writer.add_scalar('Loss/train', train_loss, 0)
            #     self.tensorboard_writer.add_scalar('Loss/class', class_loss, 0)
            #     self.tensorboard_writer.add_scalar('Loss/mask', mask_loss, 0)
            for (train_loss, class_loss, mask_loss, val_class_loss, val_class_acc, val_mask_loss, val_mask_acc) in zip(self.train_loss,
                                                           self.class_loss,
                                                           self.mask_loss,
                                                           self.val_class_loss,
                                                           self.val_class_acc,
                                                           self.val_mask_loss,
                                                           self.val_mask_acc):
                self.tensorboard_writer.add_scalar('Loss/train', train_loss, 0)
                self.tensorboard_writer.add_scalar('Loss/class', class_loss, 0)
                self.tensorboard_writer.add_scalar('Loss/mask', mask_loss, 0)
                self.tensorboard_writer.add_scalar('Loss/val_mask', val_mask_loss, 0)
                self.tensorboard_writer.add_scalar('Acc/val_mask', val_mask_acc, 0)
                self.tensorboard_writer.add_scalar('Loss/val_class', val_class_loss, 0)
                self.tensorboard_writer.add_scalar('Acc/val_class', val_class_acc, 0)
            print('Logged previous data into tensorboard')

    def predict_sample(self):
        """
        Takes in a single batch from the vlaidation loader and performs inference.
        Plots the result with the original image, mask, label and predicted mask 
        and label
        """
        val_iter = iter(self.val_loader)
        image, mask, label = next(val_iter)

        predicted_mask, predicted_label = self.model(image.to(self.device))

        predicted_mask = predicted_mask.squeeze().detach().cpu().numpy()
        
        f, axarr = plt.subplots(self.batch_size, 3, figsize=(10, self.batch_size*4))
        if self.batch_size == 1:
            axarr[0].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            axarr[0].set_title(f'Generated image: {IDX2LABELS[label.item()]}')
            axarr[1].imshow(image.squeeze().numpy())
            axarr[1].set_title(f'Generated mask: {IDX2LABELS[label.item()]}')
            axarr[2].imshow(predicted_mask)
            # axarr[2].set_title(f'Predicted mask: {IDX2LABELS}')
        else:
            for i in range(self.batch_size):
                axarr[i, 0].imshow(image[i].permute(1, 2, 0).cpu().detach().numpy())
                axarr[i, 0].set_title(f'Generated image: {IDX2LABELS[label[i].item()]}')
                axarr[i, 1].imshow(mask[i].squeeze().numpy())
                axarr[i, 1].set_title(f'Generated mask: {IDX2LABELS[label[i].item()]}')
                axarr[i, 2].imshow(predicted_mask[i])
                # axarr[i, 2].set_title(f'Generated mask: {IDX2LABELS[label[i].item()]}')

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

        with torch.no_grad():
            for image, mask, label in self.val_loader:
                image, mask, label = image.to(self.device), mask.to(self.device), label.to(self.device)

                predicted_mask, predicted_class = self.model(image)

                running_mask_loss += self.criterion_mask(predicted_mask.squeeze().float(),
                                                        mask.squeeze().float())
                running_class_loss += self.criterion_class(predicted_class, label)

                _, indices = torch.max(predicted_class, dim=1)

                running_class_accuracy += (indices.squeeze() == label.squeeze()).sum().item()
                running_mask_accuracy += self.dice_coefficient(predicted_mask.squeeze().float(),
                                                            mask.squeeze().float())
        
        self.model.train()

        return running_mask_loss/ len(self.val_loader), running_mask_accuracy/ len(self.val_loader), running_class_loss/ len(self.val_loader), running_class_accuracy/ len(self.val_loader)

    def train(self, n_epochs):
        """
        Train the model. The model will be trained and the metrics will be saved in the 'training_logs' directory.
        Args:
            n_epochs: The number of epochs for which to train.
        """

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
                    print(f'Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss/(self.print_every)} ' \
                        f'Class loss: {class_running_loss/(self.print_every)}, Mask loss: {mask_running_loss/(self.print_every)}')

                    self.train_loss.append(running_loss/self.print_every)
                    self.class_loss.append(class_running_loss/self.print_every)
                    self.mask_loss.append(mask_running_loss/self.print_every)

                    if self.use_tensorboard:
                        self.tensorboard_writer.add_scalar('Loss/train', running_loss/self.print_every, 0)
                        self.tensorboard_writer.add_scalar('Loss/class', class_loss/self.print_every, 0)
                        self.tensorboard_writer.add_scalar('Loss/mask', mask_loss/self.print_every, 0)

                    running_loss = 0
                    mask_running_loss = 0
                    class_running_loss = 0

                    self.save_model()
            # Validate after every epoch
            val_mask_loss, val_mask_acc, val_class_loss, val_class_acc = self.validate()
            print(f'Validation Stats:\nMask loss: {val_mask_loss}, Mask accuracy: {val_mask_acc}, ' \
                    f'Class loss: {val_class_loss}, Class_accuracy: {val_class_acc}\n')
            
            # Call the learning rate scheduler
            self.scheduler.step(val_mask_loss)

            self.val_class_loss.append(val_class_loss)
            self.val_mask_loss.append(val_mask_loss)
            self.val_class_acc.append(val_class_acc)
            self.val_mask_acc.append(val_mask_acc)

            if self.use_tensorboard:
                self.tensorboard_writer.add_scalar('Loss/val_mask', val_mask_loss, 0)
                self.tensorboard_writer.add_scalar('Acc/val_mask', val_mask_acc, 0)
                self.tensorboard_writer.add_scalar('Loss/val_class', val_class_loss, 0)
                self.tensorboard_writer.add_scalar('Acc/val_class', val_class_acc, 0)