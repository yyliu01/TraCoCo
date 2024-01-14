from Utils.losses import *
from Utils.lr_scheduler import PolyLR
from validate import *


def create_model(ema=False):
    # Network definition
    net = UNet(in_channels=1, is_batchnorm=True, n_classes=2)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = numpy.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(numpy.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 40)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class NegEntropy(object):
    def __call__(self, outputs_list):
        entropy_results = torch.tensor([.0]).cuda()
        for i in outputs_list:
            current_prob = torch.softmax(i, dim=0)
            current_prob = torch.clamp(current_prob, min=1e-7, max=1.0)
            entropy_results += torch.sum(current_prob.log() * current_prob, dim=0).mean()
        return entropy_results / len(outputs_list)


class Trainer:

    def __init__(self, config_,
                 train_loader,
                 valid_set, logger=None, my_wandb=None):
        super(Trainer, self).__init__()
        self.config = config_
        # labelled_num == unlabelled_num within a batch
        self.labeled_num = int(self.config.batch_size/2)

        if not config.ddp_training:
            self.model1 = create_model()
            self.model2 = create_model()
        else:
            raise NotImplementedError
        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=config.learning_rate,
                                          momentum=config.momentum, weight_decay=config.weight_decay)

        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=config.learning_rate,
                                          momentum=config.momentum, weight_decay=config.weight_decay)

        self.lr_scheduler = PolyLR(start_lr=config.learning_rate, lr_power=0.9, total_iters=config.max_iterations)

        self.iter_num_count = 0
        self.train_loader = train_loader
        self.val_set = valid_set
        self.config = config
        self.logger = logger
        self.my_wandb = my_wandb

        # specify loss function in use
        self.sup_ce_loss = F.cross_entropy
        self.sup_dice_loss = dice_loss
        self.weight_scheduler = ConsistencyWeight(config=config)
        self.saddle_reg = NegEntropy()

    @staticmethod
    def rand_bbox(size, lam=None):
        # past implementation
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = numpy.sqrt(1. - lam)
        cut_w = numpy.int(W * cut_rat)
        cut_h = numpy.int(H * cut_rat)
        cx = numpy.random.randint(size=[B, ], low=int(W / 8), high=W)
        cy = numpy.random.randint(size=[B, ], low=int(H / 8), high=H)
        bbx1 = numpy.clip(cx - cut_w // 2, 0, W)
        bby1 = numpy.clip(cy - cut_h // 2, 0, H)
        bbx2 = numpy.clip(cx + cut_w // 2, 0, W)
        bby2 = numpy.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cut_mix(self, volume=None, mask=None, gt=None):
        mix_volume = volume.clone()
        mix_target = mask.clone()
        mix_gt = gt.clone()
        u_rand_index = torch.randperm(volume.size()[0])[:volume.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox(volume.size(), lam=numpy.random.beta(4, 4))
        for i in range(0, self.config.batch_size * 2):
            if i < mix_volume.shape[0]:
                mix_volume[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    volume[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
                mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
                mix_gt[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mix_gt[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        return mix_volume, mix_target, mix_gt

    @staticmethod
    def calculate_current_acc(preds, label_batch):
        mask = (torch.softmax(preds, dim=1).max(1)[0] > config.threshold)
        y_tilde = torch.argmax(preds, dim=1)
        current_acc = (y_tilde[mask] == label_batch[mask]).sum() / len(label_batch[mask].ravel())
        return current_acc

    def train_epoch(self, model1, model2, optimizer1, optimizer2, iter_num=0):
        for batch_idx, (normal_batch, cons_batch) in enumerate(self.train_loader):
            if iter_num >= self.config.max_iterations:
                return iter_num
            else:
                iter_num = iter_num + 1
            # The original training batch: 'volume_batch[:labeled_bs]+label_batch[:labeled_bs] => labeled info'
            #                              'volume_batch[labeled_bs:] => unlabeled info'
            volume_batch, label_batch = normal_batch['image'], normal_batch['label']
            normal_range_x, normal_range_y, normal_range_z = normal_batch['x_range'], normal_batch['y_range'], \
                normal_batch['z_range']

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            cons_volume_batch, cons_label_batch = cons_batch['image'], cons_batch['label']
            cons_range_x, cons_range_y, cons_range_z = cons_batch['x_range'], cons_batch['y_range'], \
                cons_batch['z_range']

            cons_volume_batch, cons_label_batch = cons_volume_batch.cuda(), cons_label_batch.cuda()
            gt_ratio = torch.sum(label_batch == 0) / len(torch.ravel(label_batch))

            model1.train()
            model2.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            consistency_weight = self.weight_scheduler(iter_num)
            """ model1 training """
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            sup_outputs1 = model1((volume_batch + noise).clamp(min=.0, max=1.))
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            cons_outputs1 = model1((cons_volume_batch + noise).clamp(min=.0, max=1.))
            spatial_loss1 = torch.tensor([.0], device=volume_batch.device)
            collection1 = []
            for i in range(0, cons_outputs1.shape[0]):

                a = sup_outputs1[i][:, normal_range_x[0][i]:normal_range_x[1][i],
                    normal_range_y[0][i]:normal_range_y[1][i],
                    normal_range_z[0][i]:normal_range_z[1][i]]

                b = cons_outputs1[i][:, cons_range_x[0][i]:cons_range_x[1][i],
                    cons_range_y[0][i]:cons_range_y[1][i],
                    cons_range_z[0][i]:cons_range_z[1][i]]

                if not all(list(a.shape)) > 0 or not all(list(b.shape)) > 0: continue

                a = torch.clamp(a, min=1e-6, max=1.)
                b = torch.clamp(b, min=1e-6, max=1.)
                # for spatial consistency for model1
                spatial_loss1 += F.kl_div(torch.log_softmax(a, dim=1), torch.softmax(b, dim=1), reduction='none').mean()
                collection1.append(a)
                collection1.append(b)

            """ supervised part """
            # calculate the supervised loss for model1
            sup_ce_loss1 = F.cross_entropy(sup_outputs1[:self.labeled_num], label_batch[:self.labeled_num]).mean()
            outputs_soft = F.softmax(sup_outputs1, dim=1)
            # calculate the dice loss for model1
            sup_dice_loss1 = self.sup_dice_loss(outputs_soft[:self.labeled_num, 1, :, :, :],
                                                label_batch[:self.labeled_num] == 1).mean()
            # 2xHxWxD, supervised loss for model1
            loss1_sup = 0.5 * (sup_ce_loss1 + sup_dice_loss1)
            """ unsupervised part """
            # pseudo-label1
            with torch.no_grad():
                pseudo_label1 = model2(volume_batch)

            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            unsup_outputs1 = model1((volume_batch + noise).clamp(min=.0, max=1.))

            # semi-supervised loss for unlabeled data for model1; according model2's solution 
            loss1_unsup, _ = semi_crc_loss(inputs=unsup_outputs1[self.labeled_num:],
                                           targets=pseudo_label1[self.labeled_num:],
                                           threshold=config.threshold,
                                           neg_threshold=0.1,
                                           conf_mask=True)
            # entire loss for model1
            reg = self.saddle_reg(collection1)
            loss1 = loss1_sup + loss1_unsup * consistency_weight + (
                    self.config.spatial_weight * spatial_loss1 + self.config.hyp * reg) * consistency_weight

            """ model2 training """
            # spatial-consistent
            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            sup_outputs2 = model2((volume_batch + noise).clamp(min=.0, max=1.))
            noise = torch.zeros_like(cons_volume_batch).uniform_(-.05, .05)
            cons_outputs2 = model2((cons_volume_batch + noise).clamp(min=.0, max=1.))
            spatial_loss2 = torch.tensor([.0], device=volume_batch.device)
            collection2 = []
            for i in range(0, cons_outputs2.shape[0]):
                a = sup_outputs2[i][:, normal_range_x[0][i]:normal_range_x[1][i],
                    normal_range_y[0][i]:normal_range_y[1][i],
                    normal_range_z[0][i]:normal_range_z[1][i]]

                b = cons_outputs2[i][:, cons_range_x[0][i]:cons_range_x[1][i],
                    cons_range_y[0][i]:cons_range_y[1][i],
                    cons_range_z[0][i]:cons_range_z[1][i]]

                if not all(list(a.shape)) > 0 or not all(list(b.shape)) > 0: continue
                a = a.clamp(min=1e-6, max=1.)
                b = b.clamp(min=1e-6, max=1.)
                spatial_loss2 += F.kl_div(torch.log_softmax(a, dim=1), torch.softmax(b, dim=1), reduction='none').mean()
                collection2.append(a)
                collection2.append(b)

            """ supervised part """
            # calculate the supervised ce loss for model2
            sup_ce_loss2 = F.cross_entropy(sup_outputs2[:self.labeled_num], label_batch[:self.labeled_num]).mean()
            outputs_soft = F.softmax(sup_outputs2, dim=1)
            # calculate the supervised dice loss for model2
            sup_dice_loss2 = self.sup_dice_loss(outputs_soft[:self.labeled_num, 1, :, :, :],
                                                label_batch[:self.labeled_num] == 1).mean()
            loss2_sup = 0.5 * (sup_ce_loss2 + sup_dice_loss2)
            """ unsupervised part """
            # pseudo-label2
            with torch.no_grad():
                pseudo_label2 = model1(volume_batch)

            # cutmix operation
            cut_mix_input2, cut_mix_output2, label_batch = self.cut_mix(volume_batch, pseudo_label2,
                                                                        gt=label_batch)

            noise = torch.zeros_like(volume_batch).uniform_(-.05, .05)
            unsup_outputs2 = model2((cut_mix_input2 + noise).clamp(min=.0, max=1.))

            loss2_unsup, _ = semi_crc_loss(inputs=unsup_outputs2[self.labeled_num:],
                                           targets=cut_mix_output2[self.labeled_num:],
                                           threshold=config.threshold,
                                           neg_threshold=0.1,
                                           conf_mask=True)

            loss2 = loss2_sup + loss2_unsup * consistency_weight + \
                (self.config.spatial_weight * spatial_loss2 + self.config.hyp * self.saddle_reg(collection2)) * consistency_weight

            # update learning rate
            current_lr = self.lr_scheduler.get_lr(cur_iter=iter_num)
            for _, opt_group in enumerate(optimizer1.param_groups):
                opt_group['lr'] = current_lr
            for _, opt_group in enumerate(optimizer2.param_groups):
                opt_group['lr'] = current_lr

            # update gradients
            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            if batch_idx % 8 == 0:
                pred_ratio = torch.mean(torch.softmax(pseudo_label1, dim=1)[:, 0]).item()
                pseudo_label1 = torch.argmax(pseudo_label1, dim=1)
                current_acc_t = self.calculate_current_acc(preds=cut_mix_output2,
                                                           label_batch=label_batch)

                info_dict = {"model1_ce_loss": sup_ce_loss1.mean().item(),
                             "model1_dice_loss": sup_dice_loss1.mean().item(),
                             "model2_ce_loss": sup_ce_loss2.mean().item(),
                             "model2_dice_loss": sup_dice_loss2.mean().item(),
                             "model1_pseudo-acc. > {}".format(str(self.config.threshold)): current_acc_t,
                             "model1_prob_mean": pred_ratio,
                             "model1_hard_label_ratio": torch.sum(pseudo_label1 == 0) / len(torch.ravel(pseudo_label1)),
                             "model1_gt_label_ratio": gt_ratio.item(),
                             "cons. weight": consistency_weight}

                self.my_wandb.upload_wandb_info(info_dict=info_dict)

            # if iter_num % 100 == 0:
            #    self.my_wandb.produce_2d_slice(image=cut_mix_inumpyut2, pred=cut_mix_output2,
            #                                   label=label_batch)

            self.logger.info('model1 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f spatial: %f' %
                             (iter_num, loss1.item(), loss1_unsup.item(), sup_ce_loss1.item(),
                              sup_dice_loss1.item(), spatial_loss1.item()))

            self.logger.info('model2 iteration %d : loss : %f unsup_loss: %f, loss_ce: %f loss_dice: %f spatial: %f' %
                             (iter_num, loss2.item(), loss2_unsup.item(), sup_ce_loss2.item(),
                              sup_dice_loss2.item(), spatial_loss2.item()))
        return iter_num

    def validate(self, epoch, model, model_id):
        metric_record = 0.0
        self.logger.critical("{} eval time ...".format(model_id))
        model.eval()
        assert self.val_set.aug is False, ">> no augmentation for test set"
        dataloader = iter(self.val_set)
        tbar = range(len(self.val_set))
        tbar = tqdm(tbar, ncols=135, leave=True)
        for batch_idx in tbar:
            x, y = next(dataloader)
            y_tilde, y_hat = test_single_case(model, x,
                                              stride_xy=16, stride_z=16,
                                              patch_size=(self.config.patch_size[0],
                                                          self.config.patch_size[1],
                                                          self.config.patch_size[2]),
                                              num_classes=self.config.num_classes)
            if numpy.sum(y_tilde) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(numpy.array(y_tilde),
                                                         numpy.array(y[:]))

            metric_record += numpy.asarray(single_metric)

            if batch_idx == 0 and epoch > int(self.config.n_epochs * 2 / 3) and epoch % 100 == 0:
                self.my_wandb.produce_3d_gif(result=y, name="y", current_epoch=epoch)
                self.my_wandb.produce_3d_gif(result=y_tilde, name="y_tilde", current_epoch=epoch,
                                             color="blue")

        metric_record = metric_record / len(self.val_set)
        info = {"{}_dice".format(model_id): metric_record[0],
                "{}_jaccard".format(model_id): metric_record[1],
                "{}_hd95".format(model_id): metric_record[2],
                "{}_asd".format(model_id): metric_record[3]}

        self.my_wandb.upload_wandb_info(info_dict=info)
        if epoch >= int(self.config.n_epochs * .5):
            ckpt_name = str(model_id + str(info["{}_dice".format(model_id)])) + '_' + str(epoch) + '.pth'
            self.my_wandb.save_ckpt(model, ckpt_name)

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

            if self.iter_num_count >= 500 and self.iter_num_count % self.config.validate_iter == 0:
                self.validate(epoch, self.model1, model_id="model1")
                self.validate(epoch, self.model2, model_id="model2")
        return
