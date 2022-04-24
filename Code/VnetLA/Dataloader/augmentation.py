import torch
import numpy


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = numpy.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = numpy.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        
        w1 = numpy.random.randint(0, w - self.output_size[0])
        h1 = numpy.random.randint(0, h - self.output_size[1])
        d1 = numpy.random.randint(0, d - self.output_size[2])

        w1_cons = numpy.random.randint(6, w-self.output_size[0]) if w1 == 0 else w1
        h1_cons = numpy.random.randint(6, h-self.output_size[1]) if h1 == 0 else h1

        cons_start_x = numpy.random.randint(low=int(w1_cons/4),
                                            high=int(6/4 * w1_cons) if int(6/4*w1_cons) + self.output_size[0] + 1 < w else int(3/4 * w1_cons))

        cons_start_y = numpy.random.randint(low=int(h1_cons/4),
                                            high=int(6/4 * h1_cons) if int(6/4*h1_cons) + self.output_size[1] + 1 < h else int(3/4 * h1_cons))

        if d1 == 4:
            cons_start_z = numpy.random.randint(0, d - self.output_size[2])
        else:
            cons_start_z = 8-d1

        cons_image = image[cons_start_x:cons_start_x + self.output_size[0],
                           cons_start_y:cons_start_y + self.output_size[1],
                           cons_start_z:cons_start_z + self.output_size[2]]

        cons_label = label[cons_start_x:cons_start_x + self.output_size[0],
                           cons_start_y:cons_start_y + self.output_size[1],
                           cons_start_z:cons_start_z + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        assert cons_image.shape == image.shape, print(cons_image.shape, image.shape)
        assert cons_label.shape == label.shape, print(cons_label.shape, label.shape)
        return {'image': image, 'label': label, 'cons_image': cons_image, 'cons_label': cons_label, \
                'normal_range_z': [max(0, cons_start_z-d1), min(80, 80+cons_start_z-d1)], \
                'cons_range_z': [max(0, d1-cons_start_z), min(80, 80+d1-cons_start_z)], \
                'normal_range_x': [max(0, cons_start_x-w1), min(112, 112+cons_start_x-w1)],\
                'cons_range_x': [max(0, w1-cons_start_x), min(112, 112+w1-cons_start_x)],
                'normal_range_y': [max(0, cons_start_y-h1), min(112, 112+cons_start_y-h1)],\
                'cons_range_y': [max(0, h1-cons_start_y), min(112, 112+h1-cons_start_y)]}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = numpy.random.randint(0, 4)
        image = numpy.rot90(image, k)
        label = numpy.rot90(label, k)
        axis = numpy.random.randint(0, 2)
        image = numpy.flip(image, axis=axis).copy()
        label = numpy.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = numpy.clip(self.sigma * numpy.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return image, label


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = numpy.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=numpy.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(numpy.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(numpy.float32)
        cons_image = sample['cons_image']
        cons_image = cons_image.reshape(1, cons_image.shape[0], cons_image.shape[1], cons_image.shape[2]).astype(numpy.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'],
                'z_range': sample['normal_range_z']}, \
               {'image': torch.from_numpy(cons_image), 'label': torch.from_numpy(sample['cons_label']).long(),
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z']}


