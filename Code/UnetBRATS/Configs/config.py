import os
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 1337
C.repo_name = "medical_semi_seg"

""" Experiments setting """
C.patch_size = [96, 96, 96]
C.dataset = "BRATS19"
C.code_path = os.path.realpath("") + "/Code/UnetBRATS/"
C.data_path = os.path.realpath("{}/Datasets/BRATS19/data".format("."))

""" Training setting """
# trainer
C.ddp_training = False
C.batch_size = 4
C.num_workers = 4
C.shuffle = True
C.drop_last = False
C.learning_rate = 5e-2
C.threshold = 0.65
C.spatial_weight = .3
C.hyp = .05

# rampup settings (per epoch)
C.rampup_type = "sigmoid"
C.rampup_length = 40
C.rampup_start = 0

""" Model setting """
C.num_classes = 2
C.momentum = 0.9
C.weight_decay = 1e-4
C.ema_momentum = 0.99

""" Wandb setting """
os.environ['WANDB_API_KEY'] = ""
C.use_wandb = False
C.project_name = "medical_semi_seg(ct-iter)"


""" Others """
C.save_ckpt = True
C.pvc = False

""" Evaluation """
C.validate_iter = 50
