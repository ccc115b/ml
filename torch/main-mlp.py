"""使用 PyTorch 訓練 MLP 辨識 MNIST。"""

from nnlib import data, models, train

train.set_seed(42)
dataset = data.load_mnist()
model = models.MLP(hidden_sizes=(128, 64))
train.run_classifier_demo(model, dataset, epochs=5, lr=0.002)
