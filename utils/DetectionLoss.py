import torch
import torch.nn as nn


def bbox_ciou(box1, box2, eps=1e-7):
    # box = [cx, cy, w, h]
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

    c2 = cw**2 + ch**2 + eps
    rho2 = ((box2[..., 0] - box1[..., 0])**2 + (box2[..., 1] - box1[..., 1])**2)
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(box2[..., 2] / box2[..., 3]) -
                                        torch.atan(box1[..., 2] / box1[..., 3]), 2)

    with torch.no_grad():
        alpha = v / (v-iou+1+eps)

    return iou - (rho2 / c2 + v / alpha)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, grid_size=13):
        super().__init__()
        self.num_classes = num_classes
        self.S = grid_size
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):

        B = preds.shape[0]
        device = preds.device

        obj_mask = torch.zeros(B, self.S, self.S, device=device)
        noobj_mask = torch.ones(B, self.S, self.S, device=device)
        target_box = torch.zeros(B, self.S, self.S, 4, device=device)
        target_cls = torch.zeros(B, self.S, self.S, self.num_classes, device=device)

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

        # Objectness
        pred_obj = preds[..., 0]
        loss_obj = self.bce(pred_obj, obj_mask).sum() / max(B, 1)
        loss_noobj = self.bce(pred_obj, torch.zeros_like(pred_obj)) * noobj_mask
        loss_noobj = loss_noobj.sum() / max(B, 1)

        # Box (CIoU)
        pred_box = preds[..., 1:5].sigmoid()  # cx,cy,w,h

        # Tạo mask boolean để lọc các ô có chứa vật thể
        obj_bool = obj_mask.bool()

        if obj_bool.sum() > 0:
            # Chỉ trích xuất và tính toán CIoU cho các dự đoán & mục tiêu hợp lệ
            valid_pred_box = pred_box[obj_bool]
            valid_target_box = target_box[obj_bool]

            # Hàm xử lý mảng 1D thay vì toàn bộ grid 3D
            ciou = bbox_ciou(valid_pred_box, valid_target_box)
            loss_box = (1 - ciou).sum() / max(B, 1)
        else:
            loss_box = torch.tensor(0.0, device=device)

        # Classification
        pred_cls = preds[..., 5:]
        loss_cls = self.bce(pred_cls, target_cls)
        loss_cls = (loss_cls.sum(dim=-1) * obj_mask).sum() / max(B, 1)

        total = loss_obj + 0.5 * loss_noobj + 5.0 * loss_box + loss_cls

        return total, {
            'obj': loss_obj.item(),
            'noobj': loss_noobj.item(),
            'box': loss_box.item(),
            'cls': loss_cls.item()
        }

