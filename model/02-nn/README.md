# 模型訓練實作

包含 

1. mlp / cnn : MNIST 手寫數字辨識
2. gru / gpt : 小型語言模型訓練 (pretrain 訓練語料 input.txt)
3. diffustion : 散射網路，MNIST 手寫數字生成（只選 0, 1, 7 三個字訓練，速度較快）

1. [torch](./torch/) -- 使用 pytorch 實作
2. [minitorch](./minitorch) -- 自行 DIY 微型的梯度引擎
3. [numpy-nn](./numpy-nn) -- 硬寫反傳遞梯度下降公式。
