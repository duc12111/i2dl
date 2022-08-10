"""SegmentationNN"""
from re import I
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules.pooling import MaxPool2d
from torchvision import models,transforms

class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class SegmentationNN(pl.LightningModule):

    def __init__(self, training_set,validation_set,criterion,hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.training_set =training_set
        self.validaion_set = validation_set
        self.criterion=criterion
        self.upsample_nums = 4 
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        self.feature_extractors = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True).backbone
        # Freeze the pretrainend
        for param in self.feature_extractors.parameters():
          param.detach()
        self.horizontalLayerIndices=[1,3,6,16]
        backbone_output_channel= self.feature_extractors[str(len(self.feature_extractors)-1)].out_channels
        
        curr_channel=400
        self.bottleneck = nn.Sequential(
                            nn.Conv2d(backbone_output_channel, curr_channel, kernel_size=1),
                            nn.ReLU6(),
                            nn.Conv2d(curr_channel, curr_channel, kernel_size=1),
                            nn.ReLU6()
                          ).to(hparams['device'])
        self.upsample= nn.Sequential()

        self.upsample_sublayer_num = 5
        
        for i in range(self.upsample_nums):
          self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear').to(hparams['device']))
          updated_channel = curr_channel
          if i< len(self.horizontalLayerIndices)-1:
            updated_channel += self.feature_extractors[str(self.horizontalLayerIndices[-2-i])].out_channels
          self.upsample.append(nn.Conv2d(updated_channel, curr_channel//2, kernel_size=1)
          ).to(hparams['device'])
          self.upsample.append(nn.ReLU6()).to(hparams['device'])
          self.upsample.append(ConvBlock(curr_channel//2,curr_channel//2)
          ).to(hparams['device'])
          self.upsample.append(nn.ReLU6()).to(hparams['device'])
          curr_channel= curr_channel//2
        self.last = ConvBlock(curr_channel,hparams['num_classes'])
        self.transfrom = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(240),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        layeroutputs = []
        x = self.transfrom(x)
        for i in range(len(self.feature_extractors)):
          x = self.feature_extractors[str(i)](x)
          if i in self.horizontalLayerIndices:
            layeroutputs.append(x)
        
        x = layeroutputs[-1]
        x = self.bottleneck(x)

        for i in range(0,len(self.upsample),self.upsample_sublayer_num):
          x = self.upsample[i](x)
          if i//self.upsample_sublayer_num< len(self.horizontalLayerIndices)-1:
            x = torch.concat((x,layeroutputs[-2-i//self.upsample_sublayer_num]),dim=1)
          for k in range(1,self.upsample_sublayer_num):
            x = self.upsample[i+k](x)
        return self.last(x)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def training_step(self, batch, batch_idx):
        images, targets = batch


        # Perform a forward pass on the network with inputs
        preds = self.forward(images)
        # calculate the loss with the network predictions and ground truth targets
        loss = self.criterion(
                    preds,
                    targets
                )
        # # Log the accuracy and loss values to the tensorboard
        self.log('loss', loss)

        return {'loss': loss}
    def validation_step(self, batch, batch_idx):
        images, targets = batch


        # Perform a forward pass on the network with inputs
        preds = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = self.criterion(
                    preds,
                    targets         
                )
        # # Visualise the predictions  of the model
        # if batch_idx == 0:
        #     self.visualize_predictions(images, out.detach(), targets)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        self.log('val_loss', avg_loss)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validaion_set, batch_size=self.hparams['batch_size'])

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"],weight_decay =self.hparams["wd"] )
        return optim

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
