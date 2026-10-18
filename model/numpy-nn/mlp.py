"""numpy-nn：以手寫反向傳播的 MLP 訓練 MNIST 手寫數字辨識。"""

from npnn import data, models, train

dataset = data.load_mnist()
model = models.MLP(hidden_sizes=(128, 64))
train.run_classifier_demo(model, dataset, num_epochs=5, learning_rate=0.002)
