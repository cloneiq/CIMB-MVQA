import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class PseudoOrganMaskGenerator:
    def __init__(self, image_size=(384, 384), patch_size=16, min_area=5, ema_decay=0.9, cumulative_threshold=0.65):
        self.H, self.W = image_size
        self.patch_size = patch_size
        self.H_p = self.H // patch_size
        self.W_p = self.W // patch_size
        self.num_patches = self.H_p * self.W_p + 1  # +1 for the [CLS] token
        self.min_area = min_area
        self.ema_decay = ema_decay
        self.ema_importance = None
        self.cumulative_threshold = cumulative_threshold

    def get_mask(self, visual_grad, true_structure_mask=None):
        B = visual_grad.size(0)
        device = visual_grad.device
        structure_mask = torch.zeros(B, self.H, self.W).to(device)

        if self.ema_importance is None:
            self.ema_importance = torch.zeros(B, self.num_patches).to(device)  # [B, 577]

        for b_idx in range(B):
            if true_structure_mask is not None and true_structure_mask[b_idx].sum().item() > 0:
                structure_mask[b_idx] = true_structure_mask[b_idx]
                continue
            # Step 1: compute importance excluding CLS
            importance = visual_grad[b_idx][1:, :].sum(1)  # [576]
            importance = importance / (importance.sum() + 1e-6)

            # Step 2: update EMA (only for non-CLS patches)
            self.ema_importance[b_idx, 1:] = (
                self.ema_decay * self.ema_importance[b_idx, 1:] +
                (1 - self.ema_decay) * importance.detach()
            )
            self.ema_importance[b_idx, 0] = 1.0  # Ensure CLS token is always retained

            importance = self.ema_importance[b_idx, 1:]  # Use updated EMA (576)

            # Step 3: reshape to grid and smooth
            importance_grid = importance.view(self.H_p, self.W_p).cpu().numpy()
            smoothed = gaussian_filter(importance_grid, sigma=1.0)
            smoothed_tensor = torch.from_numpy(smoothed).to(device).flatten()

            # Step 4: select important regions via cumulative threshold
            values, indices = torch.sort(smoothed_tensor, descending=True)
            values = values / (values.sum() + 1e-6)
            cumsum = torch.cumsum(values, dim=0)
            mask = (cumsum <= self.cumulative_threshold).float()
            mask[0] = 1.0  # Always select at least the most important patch

            binary_mask = torch.zeros_like(smoothed_tensor)
            binary_mask.scatter_(0, indices[mask.bool()], 1.0)
            binary_mask = binary_mask.reshape(self.H_p, self.W_p)

            # Step 5: connected component filtering
            binary_mask_np = binary_mask.cpu().numpy().astype(np.uint8)
            n_labels, label_map = cv2.connectedComponents(binary_mask_np)
            final_mask = np.zeros_like(binary_mask_np)
            for lbl in range(1, n_labels):
                area = np.sum(label_map == lbl)
                if area > self.min_area:
                    final_mask[label_map == lbl] = 1

            # Step 6: upsample to [H, W]
            final_tensor = torch.tensor(final_mask, dtype=torch.float32, device=device).view(1, 1, self.H_p, self.W_p)
            upsampled = F.interpolate(final_tensor, size=(self.H, self.W), mode='nearest').squeeze()
            structure_mask[b_idx] = upsampled

        # Step 7: Downsample back to patch space
        structure_mask_patch = F.adaptive_avg_pool2d(structure_mask.unsqueeze(1), (self.H_p, self.W_p)).squeeze(1)
        structure_mask_patch = (structure_mask_patch > 0.5).float()
        structure_mask_patch = structure_mask_patch.view(B, -1)  # [B, 576]

        # Step 8: prepend [CLS] token mask (always 1)
        cls_mask = torch.ones(B, 1, device=device)  # [B, 1]
        structure_mask_patch = torch.cat([cls_mask, structure_mask_patch], dim=1)  # [B, 577]

        return structure_mask, structure_mask_patch
