import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(f'init shape is {x.shape}')
        x1 = self.conv_1(x)
        # print(f'x1 shape is {x1.shape}')
        x3 = self.conv_3(x)
        # print(f'x3 shape is {x3.shape}')
        x5 = self.conv_5(x)
        # print(f'x5 shape is {x5.shape}')
        x = torch.cat((x1, x3, x5), dim=1)
        # print(f' shape after cat is {x.shape}')
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def show_featuremap(self, x):
        feature_maps = {}

        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

            feature_maps["down_" + str(i)] = x

        x = self.bottleneck(x)

        feature_maps["bottleneck"] = x

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            feature_maps["upsample_" + str(idx // 2)] = x

            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

            feature_maps["upsample_dconv" + str(idx // 2)] = x

        return feature_maps

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return nn.Sigmoid()(self.final_conv(x))


class SegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(8, 16, 32, 64, 128), num_convs=(2, 2, 3, 3, 3)):
        super(SegNet, self).__init__()
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Encode block
        for feature, num_conv in zip(features, num_convs):
            match num_conv:
                case 2:
                    self.encode.append(DoubleConv(in_channels, feature))
                case 3:
                    self.encode.append(TripleConv(in_channels, feature))
            in_channels = feature

        # Decode block
        for feature, num_conv in reversed(list(zip(features, num_convs))[1:]):
            in_channels = feature
            match num_conv:
                case 2:
                    self.decode.append(DoubleConv(in_channels, feature // 2))
                case 3:
                    self.decode.append(TripleConv(in_channels, feature // 2))
        match num_convs[0]:
            case 2:
                self.decode.append(DoubleConv(features[0], out_channels))
            case 3:
                self.decode.append(TripleConv(features[0], out_channels))

    def device(self):
        return next(self.parameters()).device


    def forward(self, x):
        indices = []

        for layer in self.encode:
            x = layer(x)
            x, ind = self.pool(x)
            indices.append(ind)

        for layer, ind in zip(self.decode, reversed(indices)):
            x = self.unpool(x, ind)
            x = layer(x)

        return nn.Sigmoid()(x)


class MSUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(8, 16, 32, 64)):
        super(MSUnet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode
        for feature in features:
            self.downs.append(MultiConv(in_channels, feature))
            in_channels = feature

        # Decode
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = MultiConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1, 1, 0, bias=False)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return nn.Sigmoid()(self.final_conv(x))

""" TEST """
if __name__ == "__main__":
    model = UNET(in_channels=1)
    test_input = torch.rand(1, 1, 224, 224)
    output = model(test_input)
    assert test_input.shape == output.shape

