import torch.nn as nn
import timm
import torch
import segmentation_models_pytorch as smp

class SegModel(nn.Module):
    def __init__(self, pre_trained_model, drop_rate, drop_path_rate, segmentation_type='unet',pretrained=False):
        super(SegModel, self).__init__()
        
        # A number of stages used in encoder in range [3, 5]
        self.encoder_depth = 5
        # Building an encoder using Timm
        # create a model for our encoder
        # features_only = True  --> returns the tensors of all layers
        self.encoder = timm.create_model(
            pre_trained_model,
            pretrained=pretrained,
            features_only =True,
            in_chans = 3,
            drop_rate = drop_rate,
            drop_path_rate = drop_path_rate
        )
        # Generate a picture with 3 channels and 64*64 resolution
        tensor_obj = torch.rand(1,3,64,64)
        # Extract features of encoder
        output = self.encoder(tensor_obj) 
        # get input channels of encoder
        encoder_channels = [tens.shape[1] for tens in output]
        # Lenght of the decoder channel should be the same as **encoder_depth**
        decoder_channels = [256,128,64,32,16]
        if segmentation_type == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels = encoder_channels,
                decoder_channels = decoder_channels,
                n_blocks = self.encoder_depth
            )

model = SegModel('resnet18', 0., 0.)
print(model.decoder)
