import torch.nn as nn
import timm
import torch


class SegModel(nn.Module):
    def __init__(self, pre_trained_model, drop_rate, drop_path_rate,pretrained=False):
        super(SegModel, self).__init__()

        self.encoder = timm.create_model(
            pre_trained_model,
            pretrained=pretrained,
            features_only =True,
            in_chans = 3,
            drop_rate = drop_rate,
            drop_path_rate = drop_path_rate
        )

        # Do feature extraction from the penultimate layer (pre-classifier layer)
        # Generate a batch of pictures, 1 picture with 3 channels and 64*64 resolution
        tensor_obj = torch.rand(1,3,64,64)
        output = self.encoder(tensor_obj) # model(input) to get the output
        encoder_channels = [1] + [layer.shape[1] for layer in output]
        print(self.encoder.feature_info)

model = SegModel('resnet18', 0., 0.)
