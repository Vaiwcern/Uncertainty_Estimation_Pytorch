import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, norm_type='group'):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        elif norm_type == 'none':
            pass
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Unet(nn.Module):
    def __init__(self, model_type="iterative", input_channels=4, dropout_rate=0.5, norm_type='group'):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.model_type = model_type

        # Encoder
        self.conv1 = ConvBlock(input_channels, 64, dropout_rate, norm_type)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128, dropout_rate, norm_type)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256, dropout_rate, norm_type)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512, dropout_rate, norm_type)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, dropout_rate, norm_type)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = ConvBlock(1024, 512, dropout_rate, norm_type)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = ConvBlock(512, 256, dropout_rate, norm_type)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = ConvBlock(256, 128, dropout_rate, norm_type)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = ConvBlock(128, 64, dropout_rate, norm_type)

        # Output Layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward_once(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        bn = self.bottleneck(c4)

        u6 = self.up6(bn)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        return self.output_layer(c9)

    def forward(self, x):
        if self.model_type == "iterative": 
            x = x[:, :3, :, :]  # Remove feedback channel
            zero_channel = torch.zeros_like(x[:, :1, :, :])
            
            # Apply feedback for 3 iterations
            for _ in range(3):
                x_input = torch.cat([x, zero_channel], dim=1)
                zero_channel = self.forward_once(x_input)
        
        if self.model_type == "vanilla": 
            output = self.forward_once(x)
        
        return output
