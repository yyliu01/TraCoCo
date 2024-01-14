import random
import numpy
import torch


class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        return {'image': (image - image.min()) / (image.max() - image.min()), 'label': sample['label']}


class RandomBrightness(object):
    def __init__(self, region):
        self.region = region

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        scale = random.uniform(self.region[0], self.region[1])
        image = image * scale
        image = numpy.clip(image, a_min=0., a_max=1.)
        sample = {'image': image, 'label': label}
        return sample


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
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = 0
            image = numpy.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = numpy.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = numpy.random.randint(0, w - self.output_size[0])
        h1 = numpy.random.randint(0, h - self.output_size[1])
        d1 = numpy.random.randint(0, d - self.output_size[2])

        cons_start_x = numpy.random.randint(0, w1) if w1 != 0 else w1
        cons_start_y = numpy.random.randint(0, h1) if h1 != 0 else h1
        cons_start_z = numpy.random.randint(0, d1) if d1 != 0 else d1

        # no-overlap issues
        cons_start_x = cons_start_x + int(w1 / 2) if w1 - cons_start_x > self.output_size[0] else cons_start_x
        cons_start_y = cons_start_y + int(h1 / 2) if h1 - cons_start_y > self.output_size[1] else cons_start_y
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

        a = image[0 if cons_start_x < w1 else cons_start_x - w1:self.output_size[0] - (w1 - cons_start_x) if cons_start_x < w1 else self.output_size[0],
            0 if cons_start_y < h1 else cons_start_y - h1:self.output_size[1] - (h1 - cons_start_y) if cons_start_y < h1 else self.output_size[1],
            0 if cons_start_z < d1 else cons_start_z - d1:self.output_size[2] - (d1 - cons_start_z) if cons_start_z < d1 else self.output_size[2]]

        b = cons_image[
            0 if cons_start_x > w1 else w1 - cons_start_x:self.output_size[0] - (cons_start_x - w1) if cons_start_x > w1 else self.output_size[0],
            0 if cons_start_y > h1 else h1 - cons_start_y:self.output_size[1] - (cons_start_y - h1) if cons_start_y > h1 else self.output_size[1],
            0 if cons_start_z > d1 else d1 - cons_start_z:self.output_size[2] - (cons_start_z - d1) if cons_start_z > d1 else self.output_size[2]]

        assert numpy.all(numpy.equal(a, b)), "?"
        return {'image': image, 'label': label, 'cons_image': cons_image, 'cons_label': cons_label,
                'normal_range_x': [0 if cons_start_x < w1 else cons_start_x - w1,
                                   self.output_size[0] - (w1 - cons_start_x) if cons_start_x < w1 else self.output_size[0]],
                'cons_range_x': [0 if cons_start_x > w1 else w1 - cons_start_x,
                                 self.output_size[0] - (cons_start_x - w1) if cons_start_x > w1 else self.output_size[0]],
                'normal_range_y': [0 if cons_start_y < h1 else cons_start_y - h1,
                                   self.output_size[1] - (h1 - cons_start_y) if cons_start_y < h1 else self.output_size[1]],
                'cons_range_y': [0 if cons_start_y > h1 else h1 - cons_start_y,
                                 self.output_size[1] - (cons_start_y - h1) if cons_start_y > h1 else self.output_size[1]],
                'normal_range_z': [0 if cons_start_z < d1 else cons_start_z - d1,
                                   self.output_size[2] - (d1 - cons_start_z) if cons_start_z < d1 else self.output_size[2]],
                'cons_range_z': [0 if cons_start_z > d1 else d1 - cons_start_z,
                                 self.output_size[2] - (cons_start_z - d1) if cons_start_z > d1 else self.output_size[2]]}


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
        noise = numpy.clip(self.sigma * numpy.random.randn(image.shape[0], image.shape[1], image.shape[2]),
                           -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return image, label


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = numpy.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]),
                                   dtype=numpy.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(numpy.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(numpy.float32)
        label = sample['label'].astype(numpy.int32)
        cons_image = sample['cons_image']
        cons_image = cons_image.reshape(1, cons_image.shape[0], cons_image.shape[1], cons_image.shape[2]).astype(
            numpy.float32)
        cons_label = sample['cons_label'].astype(numpy.int32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'],
                'z_range': sample['normal_range_z']}, \
               {'image': torch.from_numpy(cons_image), 'label': torch.from_numpy(cons_label).long(),
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z']}
