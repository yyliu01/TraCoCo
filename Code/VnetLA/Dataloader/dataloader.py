import numpy
import random
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class LAHeart(DataLoader):
    def __init__(self, dataset, config):

        def worker_init_fn(worker_id):
            random.seed(config.seed+worker_id)
        
        self.init_kwags = {
            'dataset': dataset,
            'drop_last': config.drop_last,
            'batch_size': config.batch_size,
            'shuffle': config.shuffle,
            'num_workers': config.num_workers,
            'pin_memory': True,
            'worker_init_fn': worker_init_fn
        }

        super(LAHeart, self).__init__(**self.init_kwags)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return numpy.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield numpy.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    " Collect Datasets into fixed-length chunks or blocks "
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
