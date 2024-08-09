"""
WS-DAN models
Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightning as L
import torchmetrics

import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d
from utils import CenterLoss, batch_augment
from sklearn.metrics import roc_curve, auc

import random

__all__ = ['WSDAN_CAL']
EPSILON = 1e-6


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def pAUC(y_true, y_pred, tprThreshold=0.8):
    """
    computer Partial AUC score with TPR Threshold
    """
    # computer ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find the indices where the TPR is above the threshold
    tprAboveThr = np.where(tpr >= tprThreshold)[0]

    if len(tprAboveThr) == 0:
        return 0.0  #  all below tpr threshold

    #Extract index for ROC segment about threshold
    start = tprAboveThr[0]
    fprAboveThr = fpr[start:]
    tprAboveThr = tpr[start:] - tprThreshold

    partialAUC = auc(fprAboveThr, tprAboveThr)
    #     plotPartialAUC(fprAboveThr, tprAboveThr, partialAUC)

    return partialAUC


class CustomLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps_per_epoch, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, last_epoch)
        self.steps_per_epoch = steps_per_epoch

    def get_lr(self):
        base_rate = 0.9
        base_duration = 2.0
        it = self._step_count
        float_iter = float(it) / self.steps_per_epoch
        return [base_lr * pow(base_rate, (self.last_epoch + float_iter) / base_duration) for base_lr in self.base_lrs]


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class WSDAN_CAL(L.LightningModule):
    def __init__(self, num_classes, base_lr, steps_per_epoch, beta, M=32, net='inception_mixed_6e', pretrained=True):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.beta = beta
        self.net = net
        self.base_lr = base_lr
        self.steps_per_epoch = steps_per_epoch

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'att' in net:
            print('==> Using MANet with resnet101 backbone')
            self.features = MANet()
            self.num_features = 2048
        elif 'cspnext' in net:
            self.backbone = timm.create_model('cspresnext50', pretrained=pretrained, num_classes=num_classes)
            self.features = self.backbone.forward_features
            self.num_features = 2048
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        self.feature_center = torch.zeros(num_classes, self.M * self.num_features).cuda()

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.center_loss_fn = CenterLoss()

        self.train_raw_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)
        self.train_crop_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)
        self.train_aux_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)
        self.val_raw_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)
        self.val_aux_acc = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)

        self.val_loss = []

        logging.info(
            'WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes,
                                                                                               self.M))

    def visualize(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.forward(X)

        # Update Feature Center
        feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
        self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = self.forward(aug_images)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        loss = (self.ce_loss_fn(y_pred_raw, y) / 3. +
                self.ce_loss_fn(y_pred_aux, y_aux) * 3. / 3. +
                self.ce_loss_fn(y_pred_aug, y_aug) * 2. / 3. +
                self.center_loss_fn(feature_matrix, feature_center_batch))

        self.train_raw_acc(y_pred_raw, y)
        self.train_crop_acc(y_pred_aug, y_aug)
        self.train_drop_acc(y_pred_aux, y_aux)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_raw_acc', self.train_raw_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_crop_acc', self.train_crop_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_aux_acc', self.train_aux_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        X_m = torch.flip(X, [3])

        y_pred_raw, y_pred_aux_raw, _, attention_map = self.forward(X)
        y_pred_raw_m, y_pred_aux_raw_m, _, attention_map_m = self.forward(X_m)

        ##################################
        # Object Localization and Refinement
        ##################################

        crop_images = batch_augment(X, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
        y_pred_crop, y_pred_aux_crop, _, _ = self.forward(crop_images)

        crop_images2 = batch_augment(X, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
        y_pred_crop2, y_pred_aux_crop2, _, _ = self.forward(crop_images2)

        crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop3, y_pred_aux_crop3, _, _ = self.forward(crop_images3)

        crop_images_m = batch_augment(X_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
        y_pred_crop_m, y_pred_aux_crop_m, _, _ = self.forward(crop_images_m)

        crop_images_m2 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
        y_pred_crop_m2, y_pred_aux_crop_m2, _, _ = self.forward(crop_images_m2)

        crop_images_m3 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop_m3, y_pred_aux_crop_m3, _, _ = self.forward(crop_images_m3)

        y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
        y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
        y_pred = (y_pred + y_pred_m) / 2.

        y_pred_aux = (y_pred_aux_raw + y_pred_aux_crop + y_pred_aux_crop2 + y_pred_aux_crop3) / 4.
        y_pred_aux_m = (y_pred_aux_raw_m + y_pred_aux_crop_m + y_pred_aux_crop_m2 + y_pred_aux_crop_m3) / 4.
        y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.

        # loss
        loss = self.ce_loss_fn(y_pred, y)
        self.val_loss.append(loss.data)

        self.val_raw_acc.update(y_pred, y)
        self.val_aux_acc.update(y_pred_aux, y)
        
        return y_pred, y_pred_aux, y

    def on_validation_epoch_end(self, all_outputs):
        self.log('val_loss', np.mean(self.val_loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_raw_acc', self.val_raw_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_aux_acc', self.val_aux_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_raw_acc.reset()
        self.val_aux_acc.reset()

        y_pred, y_pred_aux, y = list(zip(*all_outputs))
        self.log('val_raw_pAUC', pAUC(y, y_pred), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_aux_pAUC', pAUC(y, y_pred_aux), on_step=True, on_epoch=True, prog_bar=True, logger=True)


# def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.base_lr)
#
#         # Define your learning rate scheduler
#         lr_scheduler = CustomLR(optimizer, self.steps_per_epoch)
#
#         return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
#
#     def adjust_learning(self, epoch, iter):
#         base_lr = self.config.learning_rate
#         base_rate = 0.9
#         base_duration = 2.0
#         lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
#         return lr
#
#     def load_state_dict(self, state_dict, strict=True):
#         model_dict = self.state_dict()
#         pretrained_dict = {k: v for k, v in state_dict.items()
#                            if k in model_dict and model_dict[k].size() == v.size()}
#
#         if len(pretrained_dict) == len(state_dict):
#             print('%s: All params loaded' % type(self).__name__)
#         else:
#             print('%s: Some params were not loaded:' % type(self).__name__)
#             not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
#             print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
#
#         model_dict.update(pretrained_dict)
#         super(WSDAN_CAL, self).load_state_dict(model_dict)
