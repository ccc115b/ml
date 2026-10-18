"""使用 NumPy tensor 版 nn.MLP 訓練 MNIST 手寫數字辨識。"""

from minitorch import common, data, nn

dataset = data.load_mnist()
model = nn.MLP(hidden_sizes=(128, 64))
common.run_classifier_demo(model, dataset, num_epochs=5, learning_rate=0.002)
