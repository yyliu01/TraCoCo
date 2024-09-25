import torch
import numpy
# from Model.Vnet.Vnet import VNet
from collections import OrderedDict


class EntireModel(torch.nn.Module):
    def __init__(self, config):
        super(EntireModel, self).__init__()
        self.network_s = VNet(n_channels=1, n_classes=config.num_classes,
                              normalization='batchnorm', has_dropout=config.drop_out)
        self.network_t = VNet(n_channels=1, n_classes=config.num_classes,
                              normalization='batchnorm', has_dropout=config.drop_out)

        for param in self.network_t.parameters():
            param.detach_()

        self.keep_rate = config.ema_momentum
        self.cons_w = ConsistencyWeight(config=config)

    def forward(self, x):

        # if not self.training:
        #     return self.network_t.decoder(self.network_t.encoder(x))

        features = self.network_s.encoder(x)
        out = self.network_s.decoder(features)
        return out

    def ema_update(self):
        student_dict = self.network_s.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.network_t.state_dict().items():
            if key in student_dict.keys():
                new_teacher_dict[key] = (
                        student_dict[key] * (1 - self.keep_rate) + value * self.keep_rate
                )
            else:
                raise Exception("{} is not found in student encoder model".format(key))
        self.network_t.load_state_dict(new_teacher_dict, strict=True)



    @staticmethod
    def gaussian(start, current, rampup_length):
        assert rampup_length >= 0
        if current == 0:
            return .0
        if current < start:
            return .0
        if current >= rampup_length:
            return 1.0
        return numpy.exp(-5 * (1 - current / rampup_length) ** 2)

    @staticmethod
    def sigmoid(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = numpy.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
    
        return float(numpy.exp(-5.0 * phase * phase))

    @staticmethod
    def linear(current, rampup_length):
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        return current / rampup_length

    @staticmethod
    def cosine(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = numpy.clip(current, 0.0, rampup_length)
        return 1 - float(.5 * (numpy.cos(numpy.pi * current / rampup_length) + 1))



