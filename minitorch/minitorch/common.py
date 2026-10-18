"""各模型共用的邏輯：softmax、Adam 優化器、語言模型與分類模型的訓練、取樣。

語言模型（GPT、GRU）只需符合同一個介面即可共用本模組：
    model.parameters                              dict[str, ndarray]
    model.block_size                              最大序列長度
    model.forward(token_ids)                      -> logits，shape (T, V)
    model.loss_and_gradients(tokens, targets)     -> (loss, gradients)
"""

import random
import numpy as np


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


class Adam:
    def __init__(self, parameters, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.parameters = parameters
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.steps = 0

    def step(self, gradients, learning_rate):
        self.steps += 1
        for name, parameter in self.parameters.items():
            self.first[name] = self.beta1 * self.first[name] + (1 - self.beta1) * gradients[name]
            self.second[name] = self.beta2 * self.second[name] + (1 - self.beta2) * gradients[name] ** 2
            first = self.first[name] / (1 - self.beta1 ** self.steps)
            second = self.second[name] / (1 - self.beta2 ** self.steps)
            parameter -= learning_rate * first / (np.sqrt(second) + self.epsilon)


def count_parameters(model):
    return sum(parameter.size for parameter in model.parameters.values())


def train_language_model(model, corpus, num_steps=1500, learning_rate=0.02,
                         min_learning_rate=0.001, log_every=150, on_step=None):
    """每步訓練一行文字，學習率線性衰減。on_step(step) 可用來做額外處理（如讓出控制權）。"""
    optimizer = Adam(model.parameters)
    for step in range(num_steps):
        tokens, targets = corpus.encode(corpus.docs[step % len(corpus.docs)])
        loss, gradients = model.loss_and_gradients(tokens, targets)
        optimizer.step(gradients, learning_rate * (1 - step / num_steps) + min_learning_rate)
        if step == 0 or (step + 1) % log_every == 0:
            print(f'步驟 {step + 1:4d} / {num_steps}，損失 {loss:.4f}')
        if on_step is not None:
            on_step(step)
    return loss


def generate(model, corpus, num_samples=12, temperature=0.6):
    """從 BOS 開始逐字取樣，遇到 BOS 或達到 block_size 就結束。"""
    sentences = []
    for _ in range(num_samples):
        token_ids, sample = [corpus.BOS], []
        for _ in range(corpus.block_size):
            logits = model.forward(token_ids)[-1] / temperature
            probabilities = softmax(logits)
            token_id = random.choices(range(corpus.vocab_size), weights=probabilities)[0]
            if token_id == corpus.BOS:
                break
            sample.append(corpus.chars[token_id])
            token_ids.append(token_id)
        sentences.append(''.join(sample))
    return sentences


def run_language_model_demo(model, corpus, title, **train_kwargs):
    """GPT / GRU 範例共用的完整流程：印資訊 → 訓練 → 生成。"""
    print(f'{corpus.describe()}，模型參數：{count_parameters(model)}')
    train_language_model(model, corpus, **train_kwargs)
    print(f'\n--- 生成的句子（{title}） ---')
    for index, sentence in enumerate(generate(model, corpus), 1):
        print(f'{index:2d}. {sentence}')


def train_classifier(model, x_train, y_train, x_test, y_test, batch_size=64, num_epochs=2,
                     learning_rate=0.003, min_learning_rate=0.0003, log_every=100, seed=42):
    """小批次訓練分類模型（MLP、CNN 共用），每輪結束印出測試準確率。"""
    import time
    optimizer = Adam(model.parameters)
    rng = np.random.default_rng(seed)
    steps_per_epoch = len(x_train) // batch_size
    num_steps = steps_per_epoch * num_epochs
    step, start = 0, time.time()
    for epoch in range(num_epochs):
        order = rng.permutation(len(x_train))
        for i in range(steps_per_epoch):
            batch = order[i * batch_size:(i + 1) * batch_size]
            loss, gradients = model.loss_and_gradients(x_train[batch], y_train[batch])
            optimizer.step(gradients, learning_rate * (1 - step / num_steps) + min_learning_rate)
            step += 1
            if step == 1 or step % log_every == 0:
                print(f'步驟 {step:5d} / {num_steps}，損失 {loss:.4f}，{time.time() - start:.0f} 秒')
        print(f'第 {epoch + 1} 輪結束：測試準確率 {model.accuracy(x_test, y_test):.2%}')


def run_classifier_demo(model, data, **train_kwargs):
    """MLP / CNN 範例共用的完整流程：印資訊 → 訓練 → 預測前 10 張測試圖。"""
    x_train, y_train, x_test, y_test = data
    print(f'訓練資料：{len(x_train)} 張，測試資料：{len(x_test)} 張，模型參數：{count_parameters(model)}')
    train_classifier(model, x_train, y_train, x_test, y_test, **train_kwargs)
    print('\n--- 前 10 張測試圖的預測 ---')
    print('預測：', model.predict(x_test[:10]).tolist())
    print('答案：', y_test[:10].tolist())
