import torch
import torch.nn.functional as F

def squared_emd_loss(logits, target_dist):
    pred_dist = F.softmax(logits, dim=-1)
    cdf_pred = torch.cumsum(pred_dist, dim=-1)
    cdf_target = torch.cumsum(target_dist, dim=-1)
    return ((cdf_pred - cdf_target) ** 2).sum(dim=-1).mean()