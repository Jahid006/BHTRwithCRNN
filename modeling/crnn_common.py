import torch.nn as nn
import torch



class ConvBlock(nn.Module):
    def __init__(self,  in_ch, out_ch, 
                 kernel_size, 
                 stride, padding, 
                 max_pooling = (0,0),
                 batch_norm= False,
                 leaky_relu = False
        ):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
        if batch_norm: 
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True))
        if max_pooling[0]:
            layers.append(nn.MaxPool2d(kernel_size=max_pooling))

        self.conv =  nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn_config):
        super(FeatureExtractor, self).__init__()

        self.channels = cnn_config['channels']
        self.kernel_sizes = cnn_config ['kernel_sizes']
        self.strides = cnn_config['strides']
        self.paddings = cnn_config['paddings']
        self.batch_normms = cnn_config['batch_normms']
        self.max_pooling = cnn_config['max_pooling']
        self.leaky_relu = cnn_config['leaky_relu']


        # self.channels = [in_channel, 32, 64, 128, 128, 256, 256, 512, 512]
        # self.kernel_sizes =         [3,   3,   3,   3,   3,   3,   3,   2]
        # self.strides =              [1,   1,   1,   1,   1,   1,   1,   1]
        # self.paddings =             [1,   1,   1,   1,   1,   1,   1,   0]
        # self.batch_normms =         [1,   1,   1,   1,   1,   1,   1,   0]
        # self.max_pooling =     [(2, 1),(2, 1),(0, 0),(2, 1),(0, 0),(0, 0), (2, 1),(0, 0)]

        
        self.feature_extractor = nn.ModuleList(
            [
                ConvBlock(
                    self.channels[i], 
                    self.channels[i+1],
                    self.kernel_sizes[i],
                    self.strides[i],
                    self.paddings[i],
                    self.max_pooling[i],
                    self.batch_normms[i],
                    self.leaky_relu
                )
            for i in range(len(self.kernel_sizes)-1)]            
        )
        self.pooler = nn.AvgPool2d((2,1))
    
    def forward(self, image):
        for l in self.feature_extractor:
            image = l(image)
        image = self.pooler(image)
        return image



class CRNN(nn.Module):
    def __init__(self, 
                img_channel, 
                num_class,
                map_to_seq_hidden=128, 
                rnn_hidden=256, 
                leaky_relu=False,
                cnn_config = dict(),
                feature_extractor_path = None,
                load_model: callable = None
            ):
        super(CRNN, self).__init__()

        self.features_extractor = FeatureExtractor(cnn_config)

        if feature_extractor_path is not None and load_model is not None:
            self.features_extractor = load_model(self.features_extractor, feature_extractor_path)


        self.pooler = self.pooler = nn.AvgPool2d((1,2))
        self.map_to_seq = nn.Linear(512 * 1, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)


    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        features = self.features_extractor(images)
        features = self.pooler(features)
        batch, channel, height, width = features.size()
        features = features.view(batch, channel * height, width)
        
        features = features.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(features)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)




if __name__=='__main__':
    import sys
    sys.path.append("/mnt/JaHiD/Zahid/RnD/BengaliGraphemeRecognition/configs")
    from config_transformer import train_config

    m = FeatureExtractor(train_config)
    z = torch.ones((4,1,32,128))

    c = CRNN(1, 2145, cnn_config= train_config)

    print(c(z).shape)

