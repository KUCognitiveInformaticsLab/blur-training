# https://qiita.com/isso_w/items/a6f4ffa6c788b64fc6ec

import cv2
import numpy as np

img = cv2.imread("/Users/sou/data/imagenet16/test/airplane/airplane_test_000.jpg")
cv2.imwrite("in.png", img)

# コントラスト
contrast = - 128

# コントラスト調整ファクター
factor = (259 * (contrast + 255)) / (255 * (259 - contrast))

# float型に変換
newImage = np.array(img, dtype="float64")

# コントラスト調整。（0以下 or 255以上）はクリッピング
newImage = np.clip((newImage - 128) * factor + 128, 0, 255)

# int型に戻す
newImage = np.array(newImage, dtype="uint8")

# 出力
cv2.imwrite(f"out_contrast{contrast}.png", newImage)
