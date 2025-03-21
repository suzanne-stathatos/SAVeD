import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt

from model.vggForLoss import mse_loss_mask, VGG_feat
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss as sigmoid_focal_loss


def weighted_bce_loss(predictions, targets, in_mask_weight=0.0, oom_weight=5.0):
    """
    Custom BCE loss with different weights for positive (inside mask) and negative (outside mask) pixels
    
    Args:
        predictions: Model predictions (B, C, H, W), values between 0-1
        targets: Ground truth masks (B, C, H, W), binary 
        in_mask_weight: Weight for pixels inside mask (reward)
        oom_weight: Weight for pixels outside mask (penalty)
    """
    # Ensure inputs are float
    predictions = predictions.float()
    targets = targets.float()
    
    # Standard binary cross entropy
    bce = F.binary_cross_entropy(predictions, targets, reduction='none')
    
    # Create weights based on target mask
    weights = torch.ones_like(targets)
    weights[targets == 1] = in_mask_weight  # Reward for correct pixels inside mask, no weight
    weights[targets == 0] = oom_weight  # Penalty for incorrect pixels outside mask
    
    # Apply weights to BCE loss
    weighted_bce = weights * bce

    # # DEBUG: visualize each of these in one plot along 3 horizontal columns
    # fig, axs = plt.subplots(1, 3)
    # # convert c, h, w to h, w, c
    # axs[0].imshow(predictions[0].detach().cpu().numpy().transpose(1, 2, 0))
    # axs[0].set_title('Predictions')
    # axs[1].imshow(targets[0].detach().cpu().numpy().transpose(1, 2, 0))
    # axs[1].set_title('Targets')
    # axs[2].imshow(weighted_bce[0].detach().cpu().numpy().transpose(1, 2, 0))
    # axs[2].set_title('Weighted BCE')
    # plt.savefig('weighted_bce.png')
    
    return weighted_bce.mean()    


class PerceptualLoss:
    def __init__(self, perc_weight):
        self.perc_weight = perc_weight
        self.loss_model = VGG_feat()
        self.loss_model = torch.nn.DataParallel(self.loss_model).cuda()
        self.loss_model = self.loss_model.eval()
        self.criterion = mse_loss_mask().cuda()

    def update_loss(self, preds, targets, loss_mask=None):
        if loss_mask is None:
            # apply loss to all pixels unless loss_mask is provided
            loss_mask = torch.ones_like(preds)
        
        # check if preds and targets are 1-channel, if so repeat to 3 channels
        if preds.shape[1] == 1:
            preds = preds.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        vgg_feat_in = self.loss_model(preds)
        vgg_feat_out = self.loss_model(targets)

        l = self.criterion(preds, targets, loss_mask)
        wl = _exp_running_avg(l.mean(), init_val=self.perc_weight[0])
        l /= wl

        loss = l.mean()

        for _i in range(0, len(vgg_feat_in)):
            _mask = F.upsample(loss_mask,
                               size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
            l = self.criterion(vgg_feat_in[_i], vgg_feat_out[_i], _mask)
            wl = _exp_running_avg(l.mean(), init_val=self.perc_weight[_i+1])
            l /= wl

            loss += l.mean()
        return loss


# x_avg: torch variable which is initialized to init_val - weight
def _exp_running_avg(x, rho=0.99, init_val=0.0):
    x_avg = init_val

    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    return x_new


class DenoiserLoss(nn.Module):
    def __init__(self, loss_type="mse", loss_lambda=0.5, in_mask_weight=1.0, oom_weight=1.0, perc_weight=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0]):
        super().__init__()
        self.loss_type = loss_type
        self.loss_lambda = loss_lambda
        self.in_mask_weight = in_mask_weight
        self.oom_weight = oom_weight
        self.perc_weight = perc_weight
        if loss_type == "perceptual":
            self.train_perceptual_loss = PerceptualLoss(self.perc_weight)
            self.val_perceptual_loss = PerceptualLoss(self.perc_weight)

        
    def forward(self, pred, target, mask=None):
        if self.loss_type == "l1_and_obj":
            return self._compute_l1_and_obj_loss(pred, target, mask)
        elif self.loss_type == "perceptual":
            return self._compute_perceptual_loss(pred, target)
        elif self.loss_type == "mse_and_focal":
            return self._compute_mse_and_focal_loss(pred, target)
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="mean")
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="sum")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _compute_l1_and_obj_loss(self, pred, target, mask):
        self._validate_inputs(pred, target, mask)
        l1_loss = F.mse_loss(pred, target)
        
        obj_pixels = mask.sum()
        if obj_pixels > 0:
            pred = torch.sigmoid(pred)
            obj_loss = weighted_bce_loss(
                pred, mask, 
                in_mask_weight=self.in_mask_weight, 
                oom_weight=self.oom_weight
            )
            obj_loss = torch.tensor(obj_loss, device=pred.device)
        else:
            obj_loss = torch.tensor(0.0, device=pred.device)
            
        return l1_loss * self.loss_lambda + obj_loss * (1 - self.loss_lambda)
    
    def _compute_perceptual_loss(self, pred, target):
        return self.train_perceptual_loss.update_loss(pred, target)
    
    def _compute_mse_and_focal_loss(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        focal_loss = sigmoid_focal_loss(
            inputs=pred, 
            targets=target, 
            reduction="mean"
        )
        return mse_loss * self.loss_lambda + focal_loss * (1 - self.loss_lambda)
    
    def _validate_inputs(self, pred, target, mask):
        assert torch.all(target >= 0) and torch.all(target <= 1), "Target is not normalized 0-1"
        assert torch.all(mask >= 0) and torch.all(mask <= 1), "Mask is not normalized 0-1"