import os
from torchvision import transforms
from torchvision.io.image import read_image
from torch.utils.data import Dataset, DataLoader

trans_real = transforms.Compose([
  transforms.Grayscale(),
])

class RealDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_paths = os.listdir(root)
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = read_image(os.path.join(self.root, image_path)).type(torch.FloatTensor)
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':
  REAL_DIR = 'real'
  BATCH_SIZE = 256
  real_ds = RealDataset(root=REAL_DIR, transform=trans_real)
  real_dl = DataLoader(real_ds, batch_size=BATCH_SIZE)