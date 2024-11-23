import torch
import numpy
import random
import cv2
from PIL import ImageEnhance, ImageFilter, Image
from scipy.ndimage.interpolation import zoom
from scipy import ndimage

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)


class RandomColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, sample):
        img = sample['image']
        out = Image.fromarray((img * 255).astype(numpy.uint8))
        rand_num = numpy.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        out = numpy.array(out).astype(float) / 255.0
        sample.update({'image': out})
        return sample


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


class RandomZoom(object):
    def __init__(self, region):
        self.region = region

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        self.output_size = image.shape
        scale = random.uniform(self.region[0], self.region[1])
        w_, h_ = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, dsize=(h_, w_), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, dsize=(h_, w_), interpolation=cv2.INTER_NEAREST)
        sample = {'image': image, 'label': label}
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    @ staticmethod
    def random_rot_flip(image, label):
        k = numpy.random.randint(0, 4)
        image = numpy.rot90(image, k)
        label = numpy.rot90(label, k)
        axis = numpy.random.randint(0, 2)
        image = numpy.flip(image, axis=axis).copy()
        label = numpy.flip(label, axis=axis).copy()
        return image, label

    @ staticmethod
    def random_rotate(image, label):
        angle = numpy.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    @ staticmethod
    def blur(img, p=0.5):
        img = Image.fromarray((img * 255).astype(numpy.uint8))
        if random.random() < p:
            sigma = numpy.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = numpy.array(img).astype(float) / 255.0
        return img

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)

        image = self.blur(image)
        image = image.astype(numpy.float32)
        label = label.astype(numpy.uint8)
        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, resolution_size):
        self.output_size = output_size
        self.resolution_size = resolution_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # the data in ACDC has lower resolution and needed to be padding for all of them!
        # we set random here to perform the padding or skipping for TraCo.
        if random.random() < 0.15:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            image = numpy.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = numpy.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

            w, h = image.shape
            w1 = numpy.random.randint(0, w - self.output_size[0])
            h1 = numpy.random.randint(0, h - self.output_size[1])

            cons_start_x = numpy.random.randint(0, w1) if w1 != 0 else w1
            cons_start_y = numpy.random.randint(0, h1) if h1 != 0 else h1

            # no-overlap issues
            cons_start_x = cons_start_x + int(w1/2) if w1 - cons_start_x > 256 else cons_start_x
            cons_start_y = cons_start_y + int(h1/2) if h1 - cons_start_y > 256 else cons_start_y
            cons_image = image[cons_start_x:cons_start_x + self.output_size[0],
                               cons_start_y:cons_start_y + self.output_size[1]]

            cons_label = label[cons_start_x:cons_start_x + self.output_size[0],
                               cons_start_y:cons_start_y + self.output_size[1]]

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

            assert cons_image.shape == image.shape, print(cons_image.shape, image.shape)
            assert cons_label.shape == label.shape, print(cons_label.shape, label.shape)
            a = image[0 if cons_start_x < w1 else cons_start_x-w1:self.output_size[0]-(w1-cons_start_x) if cons_start_x < w1 else self.output_size[0],
                0 if cons_start_y < h1 else cons_start_y-h1:self.output_size[1]-(h1-cons_start_y) if cons_start_y < h1 else self.output_size[1]]

            b = cons_image[0 if cons_start_x > w1 else w1-cons_start_x:self.output_size[0]-(cons_start_x-w1) if cons_start_x > w1 else self.output_size[0],
                0 if cons_start_y > h1 else h1-cons_start_y:self.output_size[1]-(cons_start_y-h1) if cons_start_y > h1 else self.output_size[1]]
            assert numpy.all(numpy.equal(a, b)), "?"
        else:
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            x, y = image.shape
            cons_image = image.copy()
            cons_label = label.copy()
            w1, h1 = x, y
            cons_start_x = w1
            cons_start_y = h1

        return {'image': image, 'label': label, 'cons_image': cons_image, 'cons_label': cons_label,
                'normal_range_x': [0 if cons_start_x < w1 else cons_start_x-w1, self.resolution_size[0]-(w1-cons_start_x) if cons_start_x < w1 else self.resolution_size[0]],
                'cons_range_x': [0 if cons_start_x > w1 else w1-cons_start_x, self.resolution_size[1]-(cons_start_x-w1) if cons_start_x > w1 else self.resolution_size[1]],
                'normal_range_y': [0 if cons_start_y < h1 else cons_start_y-h1, self.resolution_size[0]-(h1-cons_start_y) if cons_start_y < h1 else self.resolution_size[0]],
                'cons_range_y': [0 if cons_start_y > h1 else h1-cons_start_y, self.resolution_size[1]-(cons_start_y-h1) if cons_start_y > h1 else self.resolution_size[1],]}


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


class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        return{'image': (image - image.min()) / (image.max()-image.min()), 'label': sample['label']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        cons_image = sample['cons_image']
        return {'image': torch.from_numpy(image).unsqueeze(0),
                'label': torch.from_numpy(sample['label']).long(),
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y']}, \
               {'image': torch.from_numpy(cons_image).unsqueeze(0), 'label': torch.from_numpy(sample['cons_label']).long(),
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y']}


