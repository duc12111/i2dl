"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules.pooling import MaxPool2d
from torchvision import models,transforms

class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
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
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        self.feature_extractors = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
        self.feature_extractors.classifier = models.segmentation.lraspp.LRASPPHead(40,960,hparams['num_classes'],128)




        # self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).features
        # # self.encoder = models.alexnet(pretrained=True).features
        # self.decoder = torch.nn.Sequential(
        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(1280, 128),
        #   ConvBlock(128, 128),



        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(128, 128),


        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(128, 128),

        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(128, 64),
        #   ConvBlock(64, 64),


           
        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(64, 48),
        #   ConvBlock(48, 32),
        #   ConvBlock(32, 32),


          
        #   nn.Upsample(scale_factor=2, mode='bilinear'),
        #   ConvBlock(32, 32),
        #   ConvBlock(32, num_classes)

        # )
         
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
        x = self.transfrom(x)
        x = self.feature_extractors(x)['out']
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
