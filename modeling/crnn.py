import torch.nn as nn


class Convolution(nn.Module):
    def __init__(
            self,  in_ch, out_ch,
            kernel_size,
            stride, padding,
            max_pooling=(0, 0),
            batch_norm=False,
            leaky_relu=False
    ):
        super(Convolution, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
        if batch_norm: 
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(
            nn.LeakyReLU(0.2, inplace=True)
            if leaky_relu else nn.ReLU(inplace=True)
        )

        if max_pooling[0]:
            layers.append(nn.MaxPool2d(kernel_size=max_pooling))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channel,
        leaky_relu
    ):
        super(FeatureExtractor, self).__init__()

        self.channels = [in_channel, 32, 64, 128, 128, 256, 256, 512, 512]
        self.kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 2]
        self.strides = [1, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [1, 1, 1, 1, 1, 1, 1, 0]
        self.batch_normms = [1, 1, 1, 1, 1, 1, 1, 0]
        self.max_pooling = [
            (2, 2), (2, 1), (0, 0), (2, 1), (0, 0), (0, 0), (2, 1), (0, 0)
        ]
                             
        self.leaky_relu = leaky_relu

        self.feature_extractor = nn.ModuleList(
            [
                Convolution(
                    self.channels[i],
                    self.channels[i+1],
                    self.kernel_sizes[i],
                    self.strides[i],
                    self.paddings[i],
                    self.max_pooling[i],
                    self.batch_normms[i],
                    self.leaky_relu
                ) for i in range(len(self.kernel_sizes))
            ]
        )

    def forward(self, image):
        for layer in self.feature_extractor:
            image = layer(image)
        return image


class Crnn(nn.Module):
    def __init__(
            self,
            img_channel,
            num_class,
            map_to_seq_hidden=128,
            rnn_hidden=256,
            leaky_relu=False
    ):
        super(Crnn, self).__init__()

        self.features_extractor = FeatureExtractor(1, leaky_relu)
        self.map_to_seq = nn.Linear(512 * 1, map_to_seq_hidden)
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        features = self.features_extractor(images)
        batch, channel, height, width = features.size()
        # print(features.shape)
        features = features.view(batch, channel * height, width)
        features = features.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(features)
        # print(seq.shape)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)
