import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F
from Configs.config import config
from Model.Unet.Unet import UNet
from cc3d import connected_components
from Dataloader.dataset import BRATSDataset

"""
# https://github.com/kleinzcy/SASSnet/blob/master/code/test_util.py
def getLargestCC(segmentation):
    # from skimage.measure import label as sm_label
    labels = sm_label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC
"""


def cct(pseudo_label):
    labels_out, N = connected_components(pseudo_label, connectivity=26, return_N=True)
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        if extracted_image.sum() < 8000:
            pseudo_label[labels_out == segid] = 0
    return pseudo_label


def test_all_case(net, val_set, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                  post_process=False, visual=False):

    total_metric = 0.0
    assert val_set.aug is False, ">> no augmentation for test set"
    dataloader = iter(val_set)
    tbar = range(len(val_set))
    tbar = tqdm(tbar, ncols=135)
    for (idx, _) in enumerate(tbar):
        image, label = next(dataloader)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size,
                                                 num_classes=num_classes,
                                                 post_process=post_process)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(np.array(prediction),
                                                     np.array(label[:]))

        total_metric += np.asarray(single_metric)

        if visual:
            # import nibabel as nib
            # struggle for where to save; modify it if you need.
            raise NotImplementedError

    avg_metric = total_metric / len(val_set)
    print("|dice={:.4f}|mIoU={:.4f}|ASD={:.4f}|95HD={:.4f}|".format(avg_metric[0], avg_metric[1],
                                                                    avg_metric[3], avg_metric[2]))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1,
                     post_process=False):

    image = image.squeeze()
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda(non_blocking=True)
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if post_process:
        label_map = cct(label_map)
        # label_map = getLargestCC(label_map)  feel free to change the post-process approach

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_calculate_metric(ckpt_path, vis=False, post=False):
    net = UNet(in_channels=1, is_batchnorm=True, n_classes=2).cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()
    test_dataset = BRATSDataset(os.path.join(config.code_path, "Dataloader"),
                                config.data_path,
                                split="test", config=config)

    avg_metric = test_all_case(net, test_dataset, num_classes=2,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                               post_process=post, visual=vis)

    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medical Semi-supervised Semantic Segmentation (valid)')
    parser.add_argument("--env_name", default="traCoCo(8-label,spatial_weight(kl)=0.3,hyp=0.1,iters=9000)",
                        type=str, help="your environment folder name for training")

    parser.add_argument("--visual", action="store_true",
                        help="your environment folder name for training")

    parser.add_argument("--post", action="store_true",
                        help="implement post process or not")
    cmd_line = parser.parse_args()
    default_path = os.path.join(config.code_path, "saved", cmd_line.env_name)
    ckpt = os.listdir(default_path)
    ckpt = [i for i in ckpt if ".pth" in str(i)][0]
    print("validate {} for BRaTS19 dataset ...".format(str(ckpt)))
    metric = test_calculate_metric(os.path.join(default_path, ckpt), vis=cmd_line.visual,
                                   post=cmd_line.post)
