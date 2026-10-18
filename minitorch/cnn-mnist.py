"""使用 NumPy tensor 版 nn.CNN 訓練 MNIST 手寫數字辨識。"""

from minitorch import common, data, nn

dataset = data.load_mnist()
model = nn.CNN()
common.run_classifier_demo(model, dataset, num_epochs=2)
