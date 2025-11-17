import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, max_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2) if max_pooling is True else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.dropout:
            x = self.dropout(x)
        skip = x
        x = self.maxpool(x)
        return x, skip
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_fliter):
        super().__init__()
        
        self.down1 = DownBlock(n_channels, n_fliter)
        self.down2 = DownBlock(n_fliter, n_fliter * 2)
        self.down3 = DownBlock(n_fliter * 2, n_fliter * 4)
        self.down4 = DownBlock(n_fliter * 4, n_fliter * 8)
        self.down4 = DownBlock(n_fliter * 8, n_fliter * 16)
        
        self.bottleneck = DownBlock(n_fliter * 16, n_fliter * 32, dropout=0.4, max_pooling=False)
        
        self.up1 = UpBlock(n_fliter * 32, n_fliter * 16)
        self.up2 = UpBlock(n_fliter * 16, n_fliter * 8)
        self.up3 = UpBlock(n_fliter * 8, n_fliter * 4)
        self.up4 = UpBlock(n_fliter * 4, n_fliter * 2)
        self.up5 = UpBlock(n_fliter * 2, n_fliter)
        
        self.out_c = nn.Conv2d(n_fliter, n_fliter)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5, skip5 = self.down5(x4)
        
        x6, skip6 = self.bottleneck(x5)
        
        x = self.up1(x6, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)
        
        x = self.out_c(x)
        x = self.sigmoid(x)
        
        return x