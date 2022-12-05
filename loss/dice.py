import torch
import torch.nn as nn
import torch.nn.functional as F


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        CE_L = torch.nn.BCELoss()
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        final_dice = (0.7 * dice + 0.3 * ce) / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class WEDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(WEDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1e-5
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection, intersection_back = WEDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            if label_index == 0:
                mean = 0.004
                w = 1 / mean ** 2
                w_back = 1 / (1-mean) ** 2
            if label_index == 1:
                mean = 0.008
                w = 1 / mean ** 2
                w_back = 1 / (1-mean) ** 2
            if label_index == 2:
                mean = 0.025
                w = 1 / mean ** 2
                w_back = 1 / (1-mean) ** 2
            dice = (2 * (w * intersection + w_back * intersection_back) + smooth) / (w * (inputs.sum() + targets.sum()) + w_back * ((1-inputs).sum() + (1-targets).sum()) + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        inputs_back = 1 - inputs
        targets_back = 1 - targets
        intersection_back = torch.sum(inputs_back * targets_back)
        return intersection, intersection_back

    def forward(self, inputs, target):
        dice = 0
        # ce = 0
        # CE_L = torch.nn.BCELoss()
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            # ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        # final_dice = (1.0 * dice) / target.size(1)
        final_dice = 1.0 * dice
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices



class EDiceLoss_Val(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss_Val, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss_Val.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class MEEP_KL_Loss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(MEEP_KL_Loss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        kl = 0
        CE_L = torch.nn.BCELoss()
        KL_loss = torch.nn.KLDivLoss()
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
            kl = kl + KL_loss(F.log_softmax(inputs[:, i, ...]), F.softmax(target[:, i, ...]))

        final_dice = (0.3 * dice + 0.2 * ce + 0.5 * kl) / target.size(1)
        return final_dice

