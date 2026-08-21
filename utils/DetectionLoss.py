import torch
import torch.nn as nn


def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((box2[..., 0] - box1[..., 0]) ** 2 + (box2[..., 1] - box1[..., 1]) ** 2)
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(box2[..., 2] / box2[..., 3]) -
                                          torch.atan(box1[..., 2] / box1[..., 3]), 2)

    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    return iou - (rho2 / c2 + v / alpha)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, grid_size=13):
        super().__init__()
        self.num_classes = num_classes
        self.S = grid_size
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        B = preds.shape[0]
        device = preds.device

        obj_mask = torch.zeros(B, self.S, self.S, device=device)
        noobj_mask = torch.ones(B, self.S, self.S, device=device)
        target_box = torch.zeros(B, self.S, self.S, 4, device=device)
        target_cls = torch.zeros(B, self.S, self.S, self.num_classes, device=device)

        grid_y, grid_x = torch.meshgrid(torch.arange(self.S, device=device),
                                        torch.arange(self.S, device=device), indexing='ij')
        grid_x = grid_x.float().expand(B, -1, -1)
        grid_y = grid_y.float().expand(B, -1, -1)

        for b in range(B):
            for t in targets[b]:
                if t.sum() == 0:
                    continue

                cx, cy, w, h, cls = t.tolist()
                i = int(cx * self.S)
                j = int(cy * self.S)
                i = min(max(i, 0), self.S - 1)
                j = min(max(j, 0), self.S - 1)

                obj_mask[b, j, i] = 1
                noobj_mask[b, j, i] = 0
                target_box[b, j, i] = torch.tensor([cx, cy, w, h], device=device)
                target_cls[b, j, i, int(cls)] = 1

        # Objectness Loss (Sử dụng hệ số tiêu chuẩn của YOLO, chia cho B)
        pred_obj = preds[..., 0]

        # Thêm thủ thuật Focal Loss thủ công nhẹ nhàng để ép mô hình dập tắt Background
        bce_obj = self.bce(pred_obj, obj_mask)
        pt = torch.exp(-bce_obj)
        focal_weight = (1 - pt) ** 2.0
        focal_loss_obj = focal_weight * bce_obj

        loss_obj = (focal_loss_obj * obj_mask).sum() / B
        loss_noobj = (focal_loss_obj * noobj_mask).sum() / B

        # Box Giải mã
        pred_box_raw = preds[..., 1:5]
        pred_cx = (pred_box_raw[..., 0].sigmoid() + grid_x) / self.S
        pred_cy = (pred_box_raw[..., 1].sigmoid() + grid_y) / self.S
        pred_w = pred_box_raw[..., 2].sigmoid()
        pred_h = pred_box_raw[..., 3].sigmoid()
        pred_box_decoded = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

        # Box Loss (CIoU) - Chỉ tính trên các ô có vật thể
        obj_bool = obj_mask.bool()
        if obj_bool.sum() > 0:
            valid_pred_box = pred_box_decoded[obj_bool]
            valid_target_box = target_box[obj_bool]
            ciou = bbox_ciou(valid_pred_box, valid_target_box)
            loss_box = (1 - ciou).sum() / B
        else:
            loss_box = torch.tensor(0.0, device=device)

        # Classification Loss
        pred_cls = preds[..., 5:]
        loss_cls = (self.bce(pred_cls, target_cls).sum(dim=-1) * obj_mask).sum() / B

        # Tổ hợp Loss với trọng số chuẩn mực
        total = 1.0 * loss_obj + 0.5 * loss_noobj + 5.0 * loss_box + 1.0 * loss_cls

        return total, {
            'obj': loss_obj.item(),
            'noobj': loss_noobj.item(),
            'box': loss_box.item(),
            'cls': loss_cls.item()
        }
