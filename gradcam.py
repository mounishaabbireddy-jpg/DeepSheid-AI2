"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
for DeepShield-X Deepfake Detection Network.

Fulfills Hackathon Optional (+Bonus) Deliverable:
"8 | Grad-CAM / attention visualizations | Optional (+bonus)"

Visualizes spatial facial regions (e.g., blending boundaries, ocular asymmetry,
hairline synthesis artifacts) that the model focuses on when classifying deepfakes.
"""

import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self.model.get_last_conv_layer()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        input_tensor: [1, 3, 224, 224]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        # Target fake class logit
        score = logits[0, 0]
        score.backward(retain_graph=True)
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Interpolate to input resolution (224x224)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, torch.sigmoid(score).item()


def overlay_cam_on_image(img_np, heatmap, alpha=0.5, colormap='jet'):
    """
    Overlays Grad-CAM heatmap onto RGB image [0-255].
    """
    try:
        import cv2
        heatmap_uint8 = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        overlaid = np.float32(color_heatmap) * alpha + np.float32(img_np) * (1 - alpha)
        overlaid = np.uint8(np.clip(overlaid, 0, 255))
        return overlaid
    except ImportError:
        # Simple procedural heatmap overlay if OpenCV is not installed
        r = np.clip(heatmap * 2.0, 0.0, 1.0)
        b = np.clip(2.0 - heatmap * 2.0, 0.0, 1.0)
        g = 1.0 - np.abs(heatmap - 0.5) * 2.0
        colored = np.stack([r, g, b], axis=-1) * 255.0
        overlaid = img_np * (1.0 - alpha) + colored * alpha
        return np.uint8(np.clip(overlaid, 0, 255))

