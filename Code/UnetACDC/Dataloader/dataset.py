import h5py
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import transforms

from Dataloader.augmentation import *


class ACDCDataset(Dataset):
    """ ACDC Dataset
        input: base_dir -> your parent level path
               split -> "sup", "unsup" and "eval", must specified
    """

    def __init__(self, base_dir, data_dir,
                 split, num=None, config=None):
        self.data_dir = data_dir
        self._base_dir = base_dir
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir + '/data/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/data/test.list', 'r') as f:
                self.image_list = f.readlines()
        # https://github.com/LiheYoung/UniMatch/blob/583e32492b0ac150e0946b65864d2dcc642220b8/more-scenarios/medical/dataset/acdc.py#L32
        # we follow UniMatch to perform validation for both val & test to find the best checkpoint.
        # the final results are reported based on test set only.
        elif split == "eval":
            with open(self._base_dir + '/data/eval_test.list', 'r') as f:
                self.image_list = f.readlines()
        else:
            raise NotImplementedError

        self.image_list = [item.strip() for item in self.image_list][:-1]
        if num is not None:
            self.image_list = self.image_list[:num]
        self.aug = config.augmentation if split == 'train' else False
        self.training_transform = transforms.Compose([
            Normalise(),
            RandomColorJitter(dict(
                brightness=.5, contrast=.5,
                sharpness=.25,   color=.25
            )),
            RandomGenerator((256, 256)),
            RandomCrop((256, 256), (256, 256)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.data_dir + "/" + image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = torch.from_numpy(image).float().numpy()
        label = torch.from_numpy(numpy.array(label)).long().numpy()
        sample = {'image': image, 'label': label}
        if not self.aug:
            return sample['image'], sample['label']
        return self.training_transform(sample)


