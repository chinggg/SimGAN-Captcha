import random
from typing import Tuple
from string import ascii_uppercase, digits
from matplotlib import pyplot as plt
from wheezy.captcha import image as wheezy_captcha
from PIL import Image, ImageOps
import one_hot_encoding as ohe

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


WIDTH = 200
HEIGHT = 50
BATCH_SIZE = 128

trans_fake = transforms.Compose([
  transforms.Grayscale(),
  transforms.ToTensor()
])

text_drawings = [
            wheezy_captcha.warp(dx_factor=0.4, dy_factor=0.1),
            wheezy_captcha.rotate(angle=1),
            wheezy_captcha.offset(),
        ]
capfn = wheezy_captcha.captcha(
        drawings=[
            wheezy_captcha.background(color="#000000"),
            wheezy_captcha.text(fonts=['arial-extra.otf'], color="#FFFFFF", font_sizes=[50], drawings=text_drawings, squeeze_factor=1),
            wheezy_captcha.curve(color="#FFFFFF", number=12),
            wheezy_captcha.smooth(),
        ],
        width=WIDTH,
        height=HEIGHT,
    )

def fakecap(s='',leng=5, mode='RGB') -> Tuple[Image.Image, str]:
    if not s:
        chset = ascii_uppercase + digits
        s = ''.join(random.choices(chset, k=leng))
    im = ImageOps.invert(capfn(s))
    if mode == 'gray':
        im = im.convert('L')
    return im, s

class FakeDataset(Dataset):
    def __init__(self, size=BATCH_SIZE, transform=None):
        self.size = size
        self.transform = transform
        
    def __getitem__(self, index):
        x, label = fakecap()
        if self.transform is not None:
            x = self.transform(x)
        label = ohe.encode(label)
        return x, label
    
    def __len__(self):
        return self.size

fake_ds = FakeDataset(transform=trans_fake)
fake_dl = DataLoader(fake_ds, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    im, label = fakecap()
    plt.imshow(im, cmap='gray')
    plt.show()