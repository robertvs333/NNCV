import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights, ResNet50_Weights, resnet50


class Model(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=19
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        
        super().__init__()

        # Encoding path
        self.in_channels = in_channels
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512))

        # Decoding path
        self.up1 = (Up(1024, 256))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
# -------------------------------Second model architecture (DeepLabV3+) ----------------------------
class AtrousSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.stages = nn.ModuleList()
        
        # 1x1 conv branch
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous Separable branches
        for rate in rates:
            self.stages.append(AtrousSeparableConv(
                in_channels, out_channels, kernel_size=3, 
                padding=rate, dilation=rate
            ))
            
        # Image Pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = [stage(x) for stage in self.stages]
        
        # Global pooling branch needs upsampling
        pool = self.global_pool(x)
        pool = torch.nn.functional.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        out.append(pool)
        
        x = torch.cat(out, dim=1)
        return self.bottleneck(x)

class Decoder(nn.Module):
    """
    Decoder module for DeepLabV3 that fuses multi-scale features.
    """
    def __init__(self, aspp_channels, low_level_channels, n_classes):
        super().__init__()
        self.reduce_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            # Use the custom class we defined earlier
            # It handles Depthwise + Pointwise + BN + ReLU internally
            AtrousSeparableConv(48 + aspp_channels, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Dropout(0.5),
            AtrousSeparableConv(256, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Dropout(0.3)
        )
      
        self.out_conv = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, aspp_out, low_level_features):
        low_level_features = self.reduce_low_level(low_level_features)
        aspp_upsampled = torch.nn.functional.interpolate(aspp_out, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((aspp_upsampled, low_level_features), dim=1)
        x = self.fuse_conv(x)
        logits = self.out_conv(x)
        return logits

class DeepLabV3Plus(nn.Module):
    """ 
    A DeepLabV3+ architecture for image segmentation.
    Based on the DeepLabV3+ architecture from the original paper:
    Liang-Chieh Chen et al. (2018), "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    https://arxiv.org/pdf/1802.02611.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """
    def __init__(
        self,
        Resnet_weights=True, 
        in_channels=3, 
        n_classes=19,
        Resnet_size=101,
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        self.nclasses = n_classes
        self.in_channels = in_channels
        super().__init__()

        # Backbone (ResNet-101)
        if Resnet_weights:
            if Resnet_size == 101:
                resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            elif Resnet_size == 50:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else: 
            if Resnet_size == 101:
                resnet = resnet101(weights=None)
            elif Resnet_size == 50:
                resnet = resnet50(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for module in resnet.layer4.modules():
            if isinstance(module, nn.Conv2d):
                # If it was a downsampling layer, stop it
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                # Increase dilation to compensate for the lost stride
                # This keeps the "view" of the filters the same
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)
        self.low_level_features = nn.Sequential(*list(resnet.children())[:5])
        self.high_level_features = nn.Sequential(*list(resnet.children())[5:-2])


        # ASPP module
        self.aspp = ASPP(2048, 256,[6,12,18])  # Dilation rates of 6, 12, and 18 as in the original paper

        # Decoder
        self.decoder = Decoder(256, 256, self.nclasses)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # save input shape
        input_shape = x.shape[2:]
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        # extract low-level features from the backbone
        low_level_features = self.low_level_features(x)
        # extract high-level features from the backbone
        high_level = self.high_level_features(low_level_features)
        

        # Pass high-level features through ASPP module
        aspp_out = self.aspp(high_level)

        # Decoder
        decoder_out = self.decoder(aspp_out, low_level_features)
        # Upsample to the original input size
        logits = torch.nn.functional.interpolate(decoder_out, size=input_shape, mode='bilinear', align_corners=True)

        return logits
    

    