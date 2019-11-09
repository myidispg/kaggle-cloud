import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class FCNClassification(nn.Module):

  def __init__(self, encoder_name='efficientnet-b2', encoder_weights='imagenet',
               classes=1):
    super(FCNClassification, self).__init__()
    # Instantiate a model
    self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                     classes=classes)
    
    # The model has 2 parts: encoder and decoder. I will add a Dense Classification
    # layer on the encoder output
    self.fc1 = nn.Linear(352*10*17, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 32)
    self.fc4 = nn.Linear(32, 4)

    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    # Do a forward pass throught the encoder
    encoded_and_skip = self.model.encoder(x)

    # The encoder gives a list of the output and the skip connections.
    # First index has the output
    # Create a copy of the encoded output
    encoded_output = encoded_and_skip[0].clone()
    # print(encoded_output.shape)
    encoded_output = encoded_output.view(encoded_output.shape[0], -1)
    encoded_output = self.dropout(F.relu(self.fc1(encoded_output)))
    encoded_output = self.dropout(F.relu(self.fc2(encoded_output)))
    encoded_output = self.dropout(F.relu(self.fc3(encoded_output)))

    return self.model.decoder(encoded_and_skip), self.fc4(encoded_output)