"""使用 PyTorch 訓練 CNN 辨識 MNIST。"""

from nnlib import data, models, train

train.set_seed(42)
dataset = data.load_mnist()
model = models.CNN()
train.run_classifier_demo(model, dataset, epochs=2)
