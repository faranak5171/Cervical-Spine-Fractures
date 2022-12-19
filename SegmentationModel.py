import torch.nn as nn
import timm
import torch
import segmentation_models_pytorch as smp

'''
    cREATE A CLASS FOR SEGMENTATION MODEL

'''


class SegModel(nn.Module):
    def __init__(self, pre_trained_model, drop_rate, drop_path_rate, pretrained=False):
        super(SegModel, self).__init__()

        # A number of downsampling operations in encoder in range [3, 5]
        self.encoder_depth = 4
        num_classes = 7  # An int number of mask classification categories for C1-C7
        '''
            Building an encoder using Timm
            create a model for our encoder
            features_only = True  --> returns the tensors of all layers
            The input of a pre-trained network is (w,h,c)
        '''
        self.encoder = timm.create_model(pre_trained_model, pretrained=pretrained, features_only=True,
                                         # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                         in_chans=3,
                                         drop_rate=drop_rate,
                                         drop_path_rate=drop_path_rate)

        # Generate a picture with 3 channels and 64*64 resolution
        sample_input = torch.rand(1, 3, 64, 64)
        # Extract features learnt by encoder
        output = self.encoder(sample_input)
        '''
            Output Shape of layers in encoder [
                torch.Size([1, 64, 32, 32]), 
                torch.Size([1, 64, 16, 16]), 
                torch.Size([1, 128, 8, 8]), 
                torch.Size([1, 256, 4, 4]), 
                torch.Size([1, 512, 2, 2])]
        '''
        encoder_channels = [1] + [layer.shape[1] for layer in output]
        # default parameter for decoder channels is a list of 5 values
        decoder_channels = [256, 128, 64, 32, 16]

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels[:self.encoder_depth+1],
            decoder_channels=decoder_channels[:self.encoder_depth],
            n_blocks=self.encoder_depth)
        # Last block to produce required number of mask channels
        self.segmentation_head = nn.Conv2d(
            in_channels=decoder_channels[self.encoder_depth-1],
            out_channels=num_classes,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))

    def forward(self, x):
        global_features = [0] + self.encoder(x)[:self.encoder_depth]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
