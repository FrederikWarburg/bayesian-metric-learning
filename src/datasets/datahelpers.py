import os

import numpy as np
import torch
from PIL import Image


def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
    return (a - b + 180) % 360 - 180


def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        # with open(path, 'rb') as f:
        img = Image.open(path)
        return img.convert("RGB")
    except IOError:
        print("Black Image Used for path: ", path)
        img = Image.fromarray(np.zeros((20, 14, 3), dtype=np.uint8))
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(),
        :,
    ]
    return x.view(xsize)


def collate_tuples(batch):

    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]], [batch[0][2]]
    return (
        [batch[i][0] for i in range(len(batch))],
        [batch[i][1] for i in range(len(batch))],
        [batch[i][2] for i in range(len(batch))],
    )
