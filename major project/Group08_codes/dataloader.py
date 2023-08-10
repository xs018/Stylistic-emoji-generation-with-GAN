import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import torchvision.utils as vutils
import matplotlib.pyplot as plt


class EmojiDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.dataset = np.load(data_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx, ...]
        img = (img * 255).astype(np.uint8)

        if self.transform:
            img = self.transform(img)

        return img

class MyDataset():
    def __init__(self, data_dir, batch_size, num_workers=4):
        transforms = T.Compose([T.ToPILImage(), 
                                T.Resize(64), 
                                T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.dataset = EmojiDataset(data_dir, transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_loader(self):
        return DataLoader(self.dataset, 
                            batch_size=self.batch_size, 
                            shuffle=True, 
                            num_workers=self.num_workers,
                            pin_memory=True, 
                            drop_last=True)


if __name__ == '__main__':
    dataset = MyDataset("training_data_cartoonset10k_96_96.npy", 64)
    loader = dataset.get_loader()
    for real_batch in loader:
        plt.figure(figsize=(8,8))
        plt.imshow(np.transpose(vutils.make_grid(real_batch, padding=2, normalize=True).cpu(),(1,2,0)))
        break

    plt.show()