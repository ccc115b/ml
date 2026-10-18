"""優化器、學習率排程與梯度裁剪。"""

import math
import numpy as np


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum, self.weight_decay = momentum, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data
            v *= self.momentum
            v += grad
            p.data -= self.lr * v


class Adam(Optimizer):
    """weight_decay 以 L2 形式加進梯度；AdamW 則是「解耦」的權重衰減。"""
    decoupled = False

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.first = [np.zeros_like(p.data) for p in self.parameters]
        self.second = [np.zeros_like(p.data) for p in self.parameters]
        self.steps = 0

    def step(self):
        self.steps += 1
        for p, m, v in zip(self.parameters, self.first, self.second):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay and not self.decoupled:
                grad = grad + self.weight_decay * p.data
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2
            update = (m / (1 - self.beta1 ** self.steps)) / (np.sqrt(v / (1 - self.beta2 ** self.steps)) + self.eps)
            if self.weight_decay and self.decoupled:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * update


class AdamW(Adam):
    decoupled = True

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps, weight_decay)


def clip_grad_norm_(parameters, max_norm):
    """所有梯度合併後的 L2 範數若超過 max_norm，就等比例縮小；回傳縮放前的範數。"""
    grads = [p.grad for p in parameters if p.grad is not None]
    total = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1:
        for g in grads:
            g *= scale
    return total


class LambdaLR:
    """學習率 = 初始學習率 × lr_lambda(step)。每個優化步驟後呼叫 step()。"""

    def __init__(self, optimizer, lr_lambda):
        self.optimizer, self.lr_lambda = optimizer, lr_lambda
        self.base_lr, self.steps = optimizer.lr, 0
        optimizer.lr = self.base_lr * lr_lambda(0)

    def step(self):
        self.steps += 1
        self.optimizer.lr = self.base_lr * self.lr_lambda(self.steps)


def warmup_cosine(total_steps, warmup_steps=0, min_ratio=0.0):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


def linear_decay(total_steps, min_ratio=0.0):
    return lambda step: min_ratio + (1 - min_ratio) * max(0.0, 1 - step / total_steps)
