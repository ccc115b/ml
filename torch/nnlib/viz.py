"""把影像輸出成 PNG 或終端機 ASCII（不依賴 matplotlib / PIL）。"""

import struct
import zlib
import numpy as np


def make_grid(images, columns, pad=1):
    """images (N,H,W)，值域 0~1 → 排成一張大圖。"""
    n, h, w = images.shape
    rows = -(-n // columns)
    canvas = np.ones((rows * (h + pad) + pad, columns * (w + pad) + pad)) * 0.5
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        canvas[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = image
    return canvas


def save_png(path, image, scale=1):
    """儲存灰階 PNG；image 為 2D、值域 0~1，scale 為整數放大倍率。"""
    pixels = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pixels = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    height, width = pixels.shape
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)

    def chunk(kind, data):
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body))

    with open(path, 'wb') as file:
        file.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
                   + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def ascii_row(images, ramp=' .:-=+*#%@'):
    """把多張 (H,W)、值域 0~1 的小圖並排成終端機文字（每個像素用兩個字元寬）。"""
    lines = []
    for r in range(images.shape[1]):
        cells = [''.join(ramp[int(np.clip(v, 0, 1) * (len(ramp) - 1))] * 2 for v in image[r]) for image in images]
        lines.append('  |  '.join(cells))
    return '\n'.join(lines)
