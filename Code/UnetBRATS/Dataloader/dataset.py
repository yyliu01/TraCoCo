from Dataloader.augmentation import *
from torch.utils.data import Dataset
from torchvision import transforms
import h5py


class BRATSDataset(Dataset):
    """ BRATS Dataset
        input: base_dir -> your parent level path
               split -> "sup", "unsup" and "eval", must specified
    """
    def __init__(self, base_dir, data_dir,
                 split, num=None, config=None):
        self.data_dir = data_dir
        self._base_dir = base_dir
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir+'/data/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == "val":
            with open(self._base_dir+'/data/val.list', 'r') as f:
                self.image_list = f.readlines()
        else:
            with open(self._base_dir+'/data/test.list', 'r') as f:
                self.image_list = f.readlines()
        
        self.image_list = [item.strip() for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        self.aug = True if split == 'train' else False
        self.training_transform = transforms.Compose([
            RandomRotFlip(),
            Normalise(),
            RandomBrightness((.75, 1.25)),
            RandomCrop((config.patch_size[0],
                        config.patch_size[1],
                        config.patch_size[2])),
            ToTensor(),
        ])
        self.testing_transform = transforms.Compose([
            Normalise()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.data_dir+"/"+image_name+".h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if not self.aug:
            sample = self.testing_transform(sample)
            return sample['image'], sample['label']
        return self.training_transform(sample)


