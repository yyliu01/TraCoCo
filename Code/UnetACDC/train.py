import torch.nn.functional as F

from Utils.dice import DiceLoss
from Utils.losses import *
from Utils.lr_scheduler import PolyLR
from validate import *


def create_model(ema=False):
    # Network definition
    net = UNet(in_chns=1, class_num=4)

    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class NegEntropy(object):
    def __call__(self, outputs_list):
        entropy_results = torch.tensor([.0]).cuda()
        for i in outputs_list:
            current_prob = torch.softmax(i, dim=0)
            entropy_results += torch.sum(current_prob.log() * current_prob, dim=0).mean()
        return entropy_results / len(outputs_list)


# this fine-tuning stage follows the previous work below:
# https://github.com/DeepMed-Lab-ECNU/BCP/blob/a925e3018b23255e65a62dd34ae9ac9fc18c0bc9/code/ACDC_BCP_train.py#L89C1-L109C43
def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    mask = torch.argmax(segmentation, dim=1).detach().cpu().numpy()
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = mask[i]
            labels = label(temp_seg == c)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(labels)
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    batch_list = torch.tensor(batch_list).cuda()
    if len(torch.unique(batch_list)) == 1: return segmentation
    return torch.nn.functional.one_hot(batch_list, num_classes=4).permute(0, 3, 1, 2) * segmentation


class Trainer:
    def __init__(self, config_, train_loader, valid_set, logger=None, my_wandb=None):
        super(Trainer, self).__init__()

        self.config = config_
        # labelled_num == unlabelled_num within a batch
        self.labeled_num = int(self.config.batch_size / 2)
        if not self.config.ddp_training:
            self.model1 = create_model()
            self.model2 = create_model()
        else:
            raise NotImplementedError
        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.config.learning_rate,
                                          momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.config.learning_rate,
                                          momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.lr_scheduler = PolyLR(start_lr=self.config.learning_rate, lr_power=0.9,
                                   total_iters=self.config.max_iterations)

        self.iter_num_count = 0
        self.train_loader = train_loader
        self.val_set = valid_set
        self.logger = logger
        self.my_wandb = my_wandb

        # specify loss function in use
        self.sup_ce_loss = F.cross_entropy
        self.sup_dice_loss = DiceLoss(4)
        self.weight_scheduler = ConsistencyWeight(config=self.config)
        self.saddle_reg = NegEntropy()

    @staticmethod
    def rand_bbox(size, lam=None):
        # past implementation
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
        cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cut_mix(self, volume=None, mask=None):
        mix_volume = volume.clone()
        mix_target = mask.clone()
        u_rand_index = torch.randperm(volume.size()[0])[:volume.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox(volume.size(), lam=np.random.beta(4, 4))

        for i in range(0, mix_volume.shape[0]):
            mix_volume[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                volume[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            if len(mix_target.shape) > 3:
                mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            else:
                mix_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        return mix_volume, mix_target

    def calculate_current_acc(self, preds, label_batch):
        mask = (torch.softmax(preds, dim=1).max(1)[0] > self.config.threshold)
        y_tilde = torch.argmax(preds, dim=1)
        current_acc = (y_tilde[mask] == label_batch[mask]).sum() / len(label_batch[mask].ravel())
        return current_acc

    def train_epoch(self, model1, model2, optimizer1, optimizer2, iter_num=0):
        for batch_idx, (normal_batch, cons_batch) in enumerate(self.train_loader):
            if iter_num >= self.config.max_iterations:
                return iter_num
            else:
                iter_num = iter_num + 1

            volume_batch, label_batch = normal_batch['image'], normal_batch['label']
            cons_volume_batch, cons_label_batch = cons_batch['image'], cons_batch['label']

            normal_range_x, normal_range_y = normal_batch['x_range'], normal_batch['y_range']
            cons_range_x, cons_range_y = cons_batch['x_range'], cons_batch['y_range']

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            cons_volume_batch, cons_label_batch = cons_volume_batch.cuda(), cons_label_batch.cuda()

            model1.train()
            model2.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            consistency_weight = self.weight_scheduler(iter_num)
            """ model1 training """
            # spatial-consistent
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            sup_outputs1 = model1(volume_batch + noise)
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            cons_outputs1 = model1(cons_volume_batch + noise)
            spatial_loss1 = torch.tensor([.0], device=volume_batch.device)
            collection_1 = []
            for i in range(0, cons_outputs1.shape[0]):
                a = sup_outputs1[i][:, normal_range_x[0][i]:normal_range_x[1][i],
                    normal_range_y[0][i]:normal_range_y[1][i]]

                b = cons_outputs1[i][:, cons_range_x[0][i]:cons_range_x[1][i],
                    cons_range_y[0][i]:cons_range_y[1][i]]

                if not all(list(a.shape)) > 0 or not all(list(b.shape)) > 0: continue

                a = a.clamp(min=1e-6, max=1.)
                b = b.clamp(min=1e-6, max=1.)

                spatial_loss1 += F.kl_div(torch.log_softmax(a, dim=1), torch.softmax(b, dim=1), reduction='none').mean()
                collection_1.append(a)
                collection_1.append(b)

            """ supervised part """
            # calculate the loss
            sup_ce_loss1 = F.cross_entropy(sup_outputs1[:self.labeled_num], label_batch[:self.labeled_num]).mean()
            outputs_soft = F.softmax(sup_outputs1, dim=1)
            sup_dice_loss1 = self.sup_dice_loss(outputs_soft[:self.labeled_num],
                                                label_batch[:self.labeled_num]).mean()

            loss1_sup = 0.5 * (sup_ce_loss1 + sup_dice_loss1)

            """ unsupervised part """
            # pseudo-label
            with torch.no_grad():
                pseudo_label1 = model2(volume_batch)
                pseudo_label1 = get_ACDC_2DLargestCC(pseudo_label1)
            cut_mix_input1, cut_mix_output1 = self.cut_mix(volume_batch, pseudo_label1)

            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            unsup_outputs1 = model1(cut_mix_input1 + noise)
            loss1_cbc, _ = semi_cbc_loss(inputs=unsup_outputs1[self.labeled_num:],
                                         targets=cut_mix_output1[self.labeled_num:],
                                         threshold=config.threshold,
                                         neg_threshold=0.1,
                                         conf_mask=True)

            unsup_loss_dice_1 = self.sup_dice_loss(torch.softmax(unsup_outputs1, dim=1)[self.labeled_num:],
                                                   torch.argmax(cut_mix_output1[self.labeled_num:], dim=1)).mean()
            loss1_trans = self.config.spatial_weight * spatial_loss1 + self.config.hyp * self.saddle_reg(collection_1)
            loss1_unsup = (loss1_cbc + unsup_loss_dice_1) / 2 + loss1_trans
            loss1 = loss1_sup + loss1_unsup * consistency_weight

            # """ model2 training """

            # spatial-consistent
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            sup_outputs2 = model2(volume_batch + noise)
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            cons_outputs2 = model2(cons_volume_batch + noise)
            spatial_loss2 = torch.tensor([.0], device=volume_batch.device)
            collection_2 = []
            for i in range(0, cons_outputs2.shape[0]):
                a = sup_outputs2[i][:, normal_range_x[0][i]:normal_range_x[1][i],
                    normal_range_y[0][i]:normal_range_y[1][i]]

                b = cons_outputs2[i][:, cons_range_x[0][i]:cons_range_x[1][i],
                    cons_range_y[0][i]:cons_range_y[1][i]]
                if not all(list(a.shape)) > 0 or not all(list(b.shape)) > 0: continue
                a = a.clamp(min=1e-6, max=1.)
                b = b.clamp(min=1e-6, max=1.)
                spatial_loss2 += F.kl_div(torch.log_softmax(a, dim=1), torch.softmax(b, dim=1), reduction='none').mean()

                collection_2.append(a)
                collection_2.append(b)

            """ supervised part """
            # calculate the loss
            sup_ce_loss2 = F.cross_entropy(sup_outputs2[:self.labeled_num], label_batch[:self.labeled_num]).mean()
            outputs_soft = F.softmax(sup_outputs2, dim=1)
            sup_dice_loss2 = self.sup_dice_loss(outputs_soft[:self.labeled_num], label_batch[:self.labeled_num]).mean()
            loss2_sup = 0.5 * (sup_ce_loss2 + sup_dice_loss2)

            """ unsupervised part """
            # pseudo-label
            with torch.no_grad():
                pseudo_label2 = model1(volume_batch)
                pseudo_label2 = get_ACDC_2DLargestCC(pseudo_label2)

            cut_mix_input2, cut_mix_output2 = self.cut_mix(volume_batch, pseudo_label2)

            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            unsup_outputs2 = model2(cut_mix_input2 + noise)

            loss2_cbc, _ = semi_cbc_loss(inputs=unsup_outputs2[self.labeled_num:],
                                         targets=cut_mix_output2[self.labeled_num:],
                                         threshold=config.threshold,
                                         neg_threshold=0.1,
                                         conf_mask=True)

            unsup_loss_dice_2 = self.sup_dice_loss(torch.softmax(unsup_outputs2, dim=1)[self.labeled_num:],
                                                   torch.argmax(cut_mix_output2[self.labeled_num:], dim=1)).mean()

            # calculates the translated loss for loss2
            loss2_trans = self.config.spatial_weight * spatial_loss2 + self.config.hyp * self.saddle_reg(collection_2)

            # the over all semi-supervised loss
            loss2_unsup = (loss2_cbc + unsup_loss_dice_2) / 2 + loss2_trans
            loss2 = loss2_sup + loss2_unsup * consistency_weight

            # update 2 networks' learning rate
            current_lr = self.lr_scheduler.get_lr(cur_iter=iter_num)
            for _, opt_group in enumerate(optimizer1.param_groups):
                opt_group['lr'] = current_lr
            for _, opt_group in enumerate(optimizer2.param_groups):
                opt_group['lr'] = current_lr

            # update 2 networks' gradients
            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            self.logger.info('model1 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f spatial: %f' %
                             (iter_num, loss1.item(), loss1_cbc.item(), sup_ce_loss1.item(),
                              sup_dice_loss1.item(), spatial_loss1.item()))

            self.logger.info('model2 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f spatial: %f' %
                             (iter_num, loss2.item(), loss2_cbc.item(), sup_ce_loss2.item(),
                              sup_dice_loss2.item(), spatial_loss2.item()))
        self.logger.info('-------------------------------------------------------')
        return iter_num

    def validate(self, epoch, model, model_id):
        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        self.logger.critical("{} eval time ...".format(model_id))
        assert self.val_set.aug is False, ">> no augmentation for eval. set"

        model.eval()
        dataloader = iter(self.val_set)
        tbar = range(len(self.val_set))
        tbar = tqdm(tbar, ncols=135)
        for batch_idx in tbar:
            x, y = next(dataloader)
            y_tilde = test_single_case(model, x)
            if np.sum(y_tilde == 1) == 0:
                first_metric = 0, 0, 0, 0
            else:
                first_metric = calculate_metric_percase(y_tilde == 1, y == 1)
            if np.sum(y_tilde == 2) == 0:
                second_metric = 0, 0, 0, 0
            else:
                second_metric = calculate_metric_percase(y_tilde == 2, y == 2)

            if np.sum(y_tilde == 3) == 0:
                third_metric = 0, 0, 0, 0
            else:
                third_metric = calculate_metric_percase(y_tilde == 3, y == 3)

            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)
        metric_record = (first_total + second_total + third_total) / (3 * len(self.val_set))
        info = {"{}_dice".format(model_id): metric_record[0],
                "{}_jaccard".format(model_id): metric_record[1],
                "{}_hd95".format(model_id): metric_record[2],
                "{}_asd".format(model_id): metric_record[3]}

        self.my_wandb.upload_wandb_info(info_dict=info)
        if self.config.save_ckpt:
            self.my_wandb.save_ckpt(model, model_id + str(info["{}_dice".format(model_id)]) + '_' + str(epoch) + '.pth')

        self.logger.critical("valid | dice: {}, "
                             "jaccard: {}, "
                             "hd95: {}"
                             "asd: {}".format(metric_record[0], metric_record[1],
                                              metric_record[2], metric_record[3]))

        return

    def run(self):
        for epoch in range(0, self.config.n_epochs):
            self.iter_num_count = self.train_epoch(model1=self.model1, optimizer1=self.optimizer1,
                                                   model2=self.model2, optimizer2=self.optimizer2,
                                                   iter_num=self.iter_num_count)

            # best evaluation protocol only for ACDC dataset
            if self.iter_num_count >= 800 and epoch % 25 == 0:
                self.validate(epoch, self.model1, model_id="model1")
                self.validate(epoch, self.model2, model_id="model2")
        return
