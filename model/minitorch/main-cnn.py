"""使用 minitorch（自動微分）訓練 CNN 辨識 MNIST。"""

import minitorch as mt
from minitorch import data, models, train

mt.manual_seed(42)
dataset = data.load_mnist()
model = models.CNN()
train.run_classifier_demo(model, dataset, epochs=2)
