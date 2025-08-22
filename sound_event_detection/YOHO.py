import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary




class YOHONet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.1):
        super(YOHONet, self).__init__()

        self.num_classes = num_classes
        self.layer_defs = [
            ([3, 3], 1, 64),
            ([3, 3], 2, 128),
            ([3, 3], 1, 128),
            ([3, 3], 1, 256),
            ([3, 3], 2, 512),
            ([3, 3], 1, 512),
            ([3, 3], 2, 512),
            ([3, 3], 1, 256),
            ([3, 3], 1, 128),
        ]

        self.layer1_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(32, eps=1e-4, affine=True)

        self.blocks = nn.ModuleList()
        in_channels = 32

        for kernel_size, stride, out_channels in self.layer_defs:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

            block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels, eps=1e-4, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-4, affine=True),
                nn.ReLU(inplace=True),

                nn.Dropout2d(dropout_rate)
            )
            self.blocks.append(block)
            in_channels = out_channels

        # Final conv: [B, C, H, W] => [B, num_classes*3, 9, 1] => squeeze => [B, 9, num_classes*3]
        self.final_conv = nn.Conv1d(in_channels*4, num_classes * 3, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, 257, 40)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        # x: (B, C, 9, 1) => (B, num_classes*3, 9, 1)
        #print(x.shape)
        x=x.reshape((batch_size,x.shape[1]*x.shape[2],-1))
        #print(x.shape)
        x = self.final_conv(x)  # (B, num_classes*3, 9, 1)
        #print(x.shape)
        x = x.permute(0, 2, 1)  # => (B, 9, num_classes*3)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    model = YOHONet( num_classes=5)
    input_tensor = torch.randn(3, 64, 81)  # (batch_size, time_bins, mel_features)
    #summary(model,input_size=( 64, 81),device='cpu')
    output = model(input_tensor)
    print(output.shape)