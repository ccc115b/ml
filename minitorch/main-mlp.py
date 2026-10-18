"""使用 minitorch（自動微分）訓練 MLP 辨識 MNIST；不需要手寫反向傳播。"""

import minitorch as mt
from minitorch import data, models, train

mt.manual_seed(42)
dataset = data.load_mnist()
model = models.MLP(hidden_sizes=(128, 64))
train.run_classifier_demo(model, dataset, epochs=5, lr=0.002)
