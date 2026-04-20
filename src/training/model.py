import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # 🔹 CNN BACKBONE
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU()
        )

        # 🔹 RNN
        self.dropout = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # 🔹 CLASSIFIER
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (B, 512, 1, W)

        b, c, h, w = conv.size()
        assert h == 1, "Height must be 1"

        conv = conv.squeeze(2)       # (B, 512, W)
        conv = conv.permute(0, 2, 1) # (B, W, 512)

        # Apply robust dropout between image features and sequence learning
        conv = self.dropout(conv)

        # 🔥 RNN
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(conv)

        self.lstm2.flatten_parameters()
        out, _ = self.lstm2(out)

        # FC
        output = self.fc(out)  # (B, W, num_classes)

        return output