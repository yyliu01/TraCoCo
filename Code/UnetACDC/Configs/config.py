import os
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 1337
C.repo_name = "medical_semi_seg"


""" Experiments setting """
C.augmentation = True
C.dataset = "ACDC"
C.code_path = os.path.realpath("") + "/Code/UnetACDC/"
C.data_path = os.path.realpath("{}/Datasets/ACDC/".format("."))

""" Training setting """
# trainer
C.ddp_training = False
C.batch_size = 24
C.num_workers = 8
C.shuffle = True
C.drop_last = False
C.learning_rate = 6.5e-2
C.threshold = 0.65
C.spatial_weight = .1
C.hyp = .01

# rampup settings (per epoch)
C.rampup_type = "sigmoid"
C.rampup_length = 40
C.rampup_start = 0

""" Model setting """
C.drop_out = True  # bayesian
C.num_classes = 4
C.momentum = 0.9
C.weight_decay = 1e-4
C.ema_momentum = 0.99

""" Wandb setting """
# os.environ['WANDB_API_KEY'] = ""
C.use_wandb = False
C.project_name = "medical_semi_seg(ct-iter)"

""" Others """
C.save_ckpt = True

# best evaluation protocol is used in this benchmark
"""
# just avoid for the followers cannot reproduce the result for 1 run;
# it is 0 for all my experiments;
# feel free to check the training log for more information.
# C.last_val_epochs = 10
"""
