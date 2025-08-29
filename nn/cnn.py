import torch
import torch.nn as nn


class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim, drop_out, ls_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # 深度可分离卷积
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # 1x1 卷积（通过线性层实现）
        self.act = nn.SiLU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # 1x1 卷积
        self.gamma = nn.Parameter(ls_init_value * torch.ones((1, 1, 1, dim))) if ls_init_value > 0 else None
        self.drop = nn.Dropout(drop_out)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = x * Nx
        
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(1, 2 * dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class ImageFeatureExtractor(nn.Module):
    def __init__(self, in_channels, base_channels, num_blocks, dropout):
        super().__init__()

        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.stages = nn.ModuleList()
        for i in range(len(num_blocks)):
            stage = nn.Sequential()
            if i > 0:
                stage.add_module('downsample', DownsampleLayer(base_channels * (2 ** (i - 1))))
            
            blocks = []
            for _ in range(num_blocks[i]):
                blocks.append(ConvNeXtV2Block(base_channels * (2 ** i), dropout))
            stage.add_module('blocks', nn.Sequential(*blocks))
            self.stages.append(stage)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.head(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
    

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)


class ImageFeatureDecoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_blocks, dropout, piy, pix):
        super().__init__()

        self.final_channels = base_channels * (2 ** (len(num_blocks) - 1))
        self.feature_h, self.feature_w = piy // (2 ** (len(num_blocks) - 1)), pix // (2 ** (len(num_blocks) - 1))

        self.projection = nn.Linear(self.final_channels, self.final_channels * self.feature_h * self.feature_w)

        self.stages = nn.ModuleList()
        for i in range(len(num_blocks) - 1, -1, -1):
            stage = nn.Sequential()
            blocks = []
            for _ in range(num_blocks[i]):
                blocks.append(ConvNeXtV2Block(base_channels * (2 ** i), dropout))
            stage.add_module('blocks', nn.Sequential(*blocks))
            if i > 0:
                stage.add_module('upsample', UpsampleLayer(base_channels * (2 ** i)))
            self.stages.append(stage)

        self.head = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, z):
        x = self.projection(z)  
        x = x.view(-1, self.final_channels, self.feature_h, self.feature_w)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_blocks, dropout, piy, pix):
        super().__init__()
        self.extractor = ImageFeatureExtractor(in_channels, base_channels, num_blocks, dropout)
        self.decoder = ImageFeatureDecoder(in_channels, base_channels, num_blocks, dropout, piy, pix)

    def forward(self, x):
        return self.decoder(self.extractor(x))

    
if __name__ == '__main__':
    model = AutoEncoder(in_channels=3, base_channels=64, num_blocks=[1, 1], dropout=0.1, piy=32, pix=32)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    torch.save(model.state_dict(), "model/test_model.pth")
    print(y.shape)