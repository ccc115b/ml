"""numpy-nn：以手寫反向傳播的 CNN 訓練 MNIST 手寫數字辨識。"""

from npnn import data, models, train

dataset = data.load_mnist()
model = models.CNN()
train.run_classifier_demo(model, dataset, num_epochs=2)
