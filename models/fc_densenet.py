import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
  def __init__(self, in_features, growth_rate, bottleneck):
    super(DenseLayer, self).__init__()

    self.batch_norm1 = nn.BatchNorm2d(in_features)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_features, bottleneck*growth_rate,
                          kernel_size=1, stride=1)
    self.batch_norm2 = nn.BatchNorm2d(bottleneck*growth_rate)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(bottleneck*growth_rate, growth_rate, kernel_size=3,
                          stride=1, padding=1)
    
    self.dropout = nn.Dropout(p=0.2)
    
  def forward(self, features):
    
    new_features = self.dropout(self.conv1(self.relu1(self.batch_norm1(features))))
    new_features = self.dropout(self.conv2(self.relu2(self.batch_norm2(new_features))))
    return torch.cat([features, new_features], 1)

class DenseBlock(nn.Module):
  def __init__(self, in_channels, depth, growth_rate, bottleneck):
    """
    Construct a dense block consisting of "depth" number of conv layers.
    Each layer yields "growth rate" number of feature maps.
    Output number of feature maps: in_channels + growth_rate * depth
    """
    super(DenseBlock, self).__init__()

    self.depth = depth # Will need this in the forward block

    self.layers = nn.ModuleDict()

    for i in range(depth):
      sub_layer = nn.ModuleDict()
      self.layers[f'dense_layer_{i+1}'] = DenseLayer(in_features = in_channels + i*growth_rate,
                                                     growth_rate = growth_rate,
                                                     bottleneck = bottleneck)

  def forward(self, features):
    
    for name, layer in self.layers.items():
      features = layer(features)
    return features

class TransitionDown(nn.Module):
  def __init__(self, in_features, out_features):
    super(TransitionDown, self).__init__()
    self.layers = nn.Sequential(nn.BatchNorm2d(in_features),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_features, out_features, stride=1,
                                          kernel_size=1),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Dropout(p=0.2)
                                )

  def forward(self, features):
    return self.layers(features)

class TransitionUp(nn.Module):
  def __init__(self, in_features, out_features):
    """
    No dropout in Transition Up because zeroing out values while upscaling seems
    to be counter intuitive. The upsampled feature maps should have all the values.
    Dropout will be in subsequent DenseBlock though.
    """
    super(TransitionUp, self).__init__()
    self.layers = nn.Sequential(nn.BatchNorm2d(in_features),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_features, out_features, stride=1,
                                           kernel_size=1),
                                 nn.ConvTranspose2d(out_features, out_features, 
                                                    kernel_size=2, stride=2)
                                )

  def forward(self, features):
    return self.layers(features)

class FCDenseNet(nn.Module):

  def __init__(self, in_channels=3, num_classes=2, num_dense_blocks=5,
               growth_rate=64, dense_block_depth=4, bottleneck=4):
    super(FCDenseNet, self).__init__()

    # Number of features added by a dense block
    dense_growth = growth_rate * dense_block_depth

    self.features = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=growth_rate * 2,
                                            kernel_size=7, stride=1,
                                            padding=3),
                                   nn.BatchNorm2d(growth_rate*2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=1,
                                                padding=1)
                                   )
  
    self.encoding_layers = nn.ModuleDict()
    # Keep track of the features in the current layer/block.
    # this keeps track of the output of transition blocks and dense blocks(after concat).
    self.feature_counts = []
    layer_features = growth_rate*2 # The first conv outputs this number of features
    self.feature_counts.append(layer_features) # After first conv before dense blocks.
    for i in range(num_dense_blocks):
      self.encoding_layers[f'dense_down_{i+1}'] = DenseBlock(in_channels=layer_features,
                                                           depth=dense_block_depth,
                                                           growth_rate=growth_rate,
                                                           bottleneck=bottleneck)

      # Output feature count of dense layers will be:
      layer_features += layer_features + dense_growth
      self.feature_counts.append(layer_features) # Append output feature count of dense block
      # ^^ layer_features is added bacause output of dense blocks are concatenated 
      # with output of previous layer.

      # Use transition layers to reduce feature map size and count
      self.encoding_layers[f'transition_down_{i+1}'] = TransitionDown(in_features = layer_features,
                                                               out_features = layer_features // 2)

      layer_features //= 2 # Account for halving due to transition block.

    # Create a bridge between the encoder and decoder block.
    # Used to recognize features from the avgpooling layer in the previous transition
    # down block. Might be useful for the classification layer to be added too
    # The bridge consists of a one by one convolution and a ReLU.
    self.bridge = nn.Sequential(
        nn.Conv2d(in_channels=layer_features, out_channels=layer_features, kernel_size=1),
        nn.ReLU(),
        nn.Dropout(p=0.2)
    )

    # DECODER BLOCK 
    self.decoder_layers = nn.ModuleDict()

    # Layer features has a knowledge of how many features are there after encoding. 
    # This is used on fc layers below.
    transition_in_features = layer_features
    
    for i in range(num_dense_blocks):
      index = -(i+1) # Access the features in reverse order one by one
      self.decoder_layers[f'transition_up_{i+1}'] = TransitionUp(in_features = transition_in_features,
                                                               out_features = transition_in_features//2)
      
      dense_in_features = self.feature_counts[index] + transition_in_features//2
      self.decoder_layers[f'dense_up_{i+1}'] = DenseBlock(in_channels=dense_in_features,
                                                          depth=dense_block_depth,
                                                          growth_rate=growth_rate,
                                                          bottleneck=bottleneck)
      transition_in_features = dense_in_features + dense_growth

    # Final transposed convolution to bring the image resolution same as input
    self.final_up_conv = nn.ConvTranspose2d(transition_in_features, num_classes,
                                            kernel_size=2, stride=1, padding=1)


    # Linear layers for classification of classes
    self.fc1 = nn.Linear(layer_features*43*65, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 4)

    self.dropout = nn.Dropout(p=0.3, inplace=True)

    # Initialize with normal distribution
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight = nn.init.xavier_uniform_(m.weight)
      elif isinstance(m, nn.Linear):
        m.weight = nn.init.xavier_uniform_(m.weight)
      elif isinstance(m, nn.ConvTranspose2d):
        m.weight = nn.init.xavier_uniform_(m.weight)

  def forward(self, x):

    x= self.features(x)
    
    skip_connections = [x]

    # Encoding
    # Keep track of the previous map for concatenation
    for name, layer in self.encoding_layers.items():
      if isinstance(layer, DenseBlock):
        dense_x = layer(x)
        x = torch.cat([x, dense_x], 1)
        skip_connections.append(x)
      elif isinstance(layer, TransitionDown):
        x = layer(x)    
 
    # Bridge
    x = self.bridge(x)
    encoded = x.clone()
    encoded = encoded.view(encoded.shape[0], -1)

    # Decoding
    for i, (name, layer) in enumerate(self.decoder_layers.items()):
      if isinstance(layer, DenseBlock):
        # Concatenate skip connection for dense block
        skip_connection = skip_connections[-(i//2+1)]
        x = self.pad(x, skip_connection)
        x = torch.cat([x, skip_connection], 1)
        x = layer(x)
      else:
        x = layer(x)
    
    # Classification for the classes
    encoded = self.dropout(self.fc1(encoded))
    encoded = self.dropout(self.fc2(encoded))

    return self.final_up_conv(x), self.fc3(encoded)

  def pad(self, features, skip_connection):
    """
    Pad the features so that they are the same size as the skip connections.
    args:
      features: The features outputtted by an transition up layer which must be 
                concatenated with features from dense layers while encoding
      skip_connection: The corresponding features generated by dense block while encoding
    """
    # Since the difference in height and/or width will not be more that one, 
    # padding on only right/bottom as required. 
    height_diff = skip_connection.shape[2] - features.shape[2]
    width_diff = skip_connection.shape[3] - features.shape[3]

    # Padding of the features is required. Skip conncection must not be disturbed.
    return torch.nn.functional.pad(features, (0, width_diff, 0, height_diff))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = FCDenseNet(in_channels=3, num_classes=1, num_dense_blocks=5,
     growth_rate=16, dense_block_depth=3)
    model = model.to(device)
    print(model)