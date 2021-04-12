# Ref: https://qiita.com/kenfukaya/items/ea72a352a281abdbebd4
from PIL import Image
from PIL import ImageEnhance


if __name__ == "__main__":
    img = Image.open("/Users/sou/data/imagenet16/test/airplane/airplane_test_000.jpg")
    # img.show()

    contrast = 0

    con = ImageEnhance.Contrast(img)
    con_img = con.enhance(contrast)

    # con_img.show()
    con_img.save(f"out_contrast{contrast}.png")

    # con_img = np.asarray(con_img) / 255
    # print(np.array(con_img))
