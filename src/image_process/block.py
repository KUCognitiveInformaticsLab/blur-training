import skimage.util


def split_image_cut(img, div_v, div_h):
    h, w = img.shape[:2]
    block_h, out_h = divmod(h, div_v)
    block_w, out_w = divmod(w, div_h)
    block_shape = (block_h, block_w, 3) if len(img.shape) == 3 else (block_h, block_w)
    return skimage.util.view_as_blocks(img[: h - out_h, : w - out_w], block_shape)
