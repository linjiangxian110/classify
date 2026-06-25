import torch
import torch.nn as nn
import numpy as np
import cv2

label = ['cloudy', 'rainy', 'snowy', 'sunny']
im_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeatherCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 26 * 26, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = WeatherCNN(num_classes=len(label)).to(device)
model.load_state_dict(torch.load("./results/model_sample.pth", map_location=device))
model.eval()


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str, 天气类别标签，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    X = cv2.resize(X, (im_size, im_size))
    X = X.astype(np.float32) / 255.0
    # HWC -> CHW，加 batch 维
    X = np.transpose(X, (2, 0, 1))[np.newaxis, :, :, :]
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        prediction = model(X)
    y_predict = label[int(torch.argmax(prediction, dim=1).item())]
    return y_predict
