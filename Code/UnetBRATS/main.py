from Utils.tensor_board import Tensorboard
import os
import torch
import numpy
import random
import argparse
import datetime
from train import Trainer
from Utils.logger import *
# from dgx.download_to_pvc import *
from torch.utils.data import DataLoader
from Configs.config import config
from Dataloader.dataset import BRATSDataset
from Dataloader.dataloader import TwoStreamBatchSampler

def main(args):
    # update the default config with the args
    config.update(vars(args))

    def worker_init_fn(worker_id):
        random.seed(config.seed + worker_id)

    # if args.pvc:
    #     os.system("cp -rf /pvc/dataset/BRATS19 /workspace/")

    train_set = BRATSDataset(os.path.join(config.code_path, "Dataloader"),
                             config.data_path, split="train", config=config)

    batch_sampler = TwoStreamBatchSampler(list(range(config.labeled_num)),
                                          list(range(config.labeled_num, len(train_set))), config.batch_size,
                                          int(config.batch_size / 2))

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                              # (pin_memory=False) to avoid the warning:
                              # "Leaking Caffe2 thread-pool after fork. (function pthreadpool)"
                              num_workers=config.num_workers, pin_memory=False,
                              worker_init_fn=worker_init_fn)

    val_dataset = BRATSDataset(os.path.join(config.code_path, "Dataloader"),
                               config.data_path,
                               split="val", num=None, config=config)

    config.iter_per_epoch = len(train_loader)
    config.n_epochs = config.max_iterations // len(train_loader)
    config.unlabeled_num = len(train_set) - config.labeled_num

    logger = logging.getLogger("TraCoCo")
    logger.propagate = False
    logger.info("training with {} epochs [{} iters]".format(config.n_epochs,
                                                            config.iter_per_epoch * config.n_epochs))
    logger.warning("running time: " + datetime.datetime.now().strftime(' [%H:%M] %d/%m/%y'))
    logger.warning("supervised sample: {}, unsupervised sample: {}".format(config.labeled_num,
                                                                           config.unlabeled_num))
    logger.critical("architecture: {}, backbone: {}".format(args.architecture,
                                                            "nothing" if args.backbone is None
                                                            else args.backbone))

    tensorboard = Tensorboard(config=config)
    trainer = Trainer(config, train_loader=train_loader, valid_set=val_dataset, logger=logger,
                      my_wandb=tensorboard)
    trainer.run()
    return


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Medical Semi-supervised Semantic Segmentation')
    # network architectures
    parser.add_argument("-a", "--architecture", default='unet', type=str,
                        help="select the architecture in use")
    parser.add_argument("-b", "--backbone", default=None, type=str,
                        help="select the architecture in use")

    # experimental settings
    parser.add_argument("--labeled_num", default=25, type=int)
    parser.add_argument("--max_iterations", default=15000, type=int)
    parser.add_argument("--unsup_weight", default=1.0, type=float)
    cmd_line_var = parser.parse_args()

    # pvc
    # parser.add_argument("--pvc", action='store_true', help="use pvc or not")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    main(cmd_line_var)
