"""minitorch：以自動微分訓練條件式擴散模型（DDPM），生成手寫數字。"""

import minitorch as mt
from minitorch import data, diffusion, models, train

CLASSES = (0, 1, 7)  # 只訓練這幾個數字；改成 tuple(range(10)) 就是全部十類
PER_CLASS = 2000     # 每個類別最多取幾張
STEPS = 3000         # 訓練步數

mt.manual_seed(42)
x_train, y_train, _, _ = data.load_mnist()
images, labels = data.select_classes(x_train, y_train, CLASSES, per_class=PER_CLASS)
images = data.shrink(images, 2) * 2 - 1  # 28×28 → 14×14，值域縮放到 [-1, 1]

model = models.UNet(num_classes=len(CLASSES), channels=8)
schedule = diffusion.Diffusion(T=100)
train.run_diffusion_demo(model, schedule, images, labels, CLASSES, steps=STEPS)
