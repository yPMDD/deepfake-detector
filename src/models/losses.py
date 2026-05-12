import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for Bounding Box Regression.
    Math: 1 - (Intersection / Union)
    """
    def __init__(self, l1_weight=1.0):
        super(IoULoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        # pred, target shape: (Batch, 4) -> [x, y, w, h]
        
        # Convert to [x1, y1, x2, y2]
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 0] + pred[:, 2]
        pred_y2 = pred[:, 1] + pred[:, 3]

        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 0] + target[:, 2]
        target_y2 = target[:, 1] + target[:, 3]

        # Calculate intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        # Ensure width/height are positive
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Calculate union area
        pred_area = pred[:, 2] * pred[:, 3]
        target_area = target[:, 2] * target[:, 3]
        union_area = pred_area + target_area - inter_area + 1e-7

        # IoU
        iou = inter_area / union_area
        
        # Loss: 1 - mean(IoU)
        iou_loss = 1.0 - torch.mean(iou)
        l1 = self.l1_loss(pred, target)
        
        # Combine losses
        return iou_loss + self.l1_weight * l1
