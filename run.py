import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import save_image
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData
from PIL import Image
import base64
import io


class Pscc:
    def __init__(self, args):
        self.device_ids = [Id for Id in range(torch.cuda.device_count())]
        self.device = torch.device('cuda:0')  # Ensure we are using GPU 0

        self.args = args

        # define backbone
        self.FENet_name = 'HRNet'
        self.FENet_cfg = get_hrnet_cfg()
        self.FENet = get_seg_model(self.FENet_cfg)

        # define localization head
        self.SegNet_name = 'NLCDetection'
        self.SegNet = NLCDetection(self.args)

        # define detection head
        self.ClsNet_name = 'DetectionHead'
        self.ClsNet = DetectionHead(self.args)

        self.FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(self.FENet_name)
        self.SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(self.SegNet_name)
        self.ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(self.ClsNet_name)

        # Load network weights
        self.load_network_weights()

    def load_network_weights(self):
        # Load FENet weight
        self.FENet = self.FENet.to(self.device)
        self.FENet = nn.DataParallel(self.FENet, device_ids=self.device_ids)
        self.load_weight(self.FENet, self.FENet_checkpoint_dir, self.FENet_name)

        # Load SegNet weight
        self.SegNet = self.SegNet.to(self.device)
        self.SegNet = nn.DataParallel(self.SegNet, device_ids=self.device_ids)
        self.load_weight(self.SegNet, self.SegNet_checkpoint_dir, self.SegNet_name)

        # Load ClsNet weight
        self.ClsNet = self.ClsNet.to(self.device)
        self.ClsNet = nn.DataParallel(self.ClsNet, device_ids=self.device_ids)
        self.load_weight(self.ClsNet, self.ClsNet_checkpoint_dir, self.ClsNet_name)

    def load_weight(self, net, checkpoint_dir, name):
        weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
        net_state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        net.load_state_dict(net_state_dict)
        print('{} weight-loading succeeds'.format(name))

    def confidence_score_pixelwise(self, prob_map):
        """
        Calculate the global confidence score based on pixelwise confidence
        using the formula: 2 * |p - 0.5| for each pixel.
        Normalize the score to ensure it's between 0 and 1.

        Args:
            prob_map (np.array): Probability map of shape (H, W), with values between 0 and 1.

        Returns:
            float: Global confidence score between 0 and 1 (1: high confidence, 0: uncertain).
        """
        # Compute pixelwise confidence: p=0 or 1 -> 1, p=0.5 -> 0
        pixel_confidences = 2 * np.abs(prob_map - 0.5)

        # Normalize the global confidence score to be between 0 and 1
        global_confidence = np.mean(pixel_confidences)

        # Scale the score to be in the range [0, 1]
        return np.clip(global_confidence, 0, 1)

    def predict(self, image: torch.Tensor):
        # Ensure image is float and normalized
        image = image.float() / 255.0

        # Ensure image has correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        if image.shape[1] == 1:  # Convert grayscale to RGB
            image = image.repeat(1, 3, 1, 1)

        image = image.to(self.device)

        # Set models to evaluation mode
        self.FENet.eval()
        self.SegNet.eval()
        self.ClsNet.eval()

        with torch.no_grad():
            # Backbone network (HRNet)
            feat = self.FENet(image)

            # Localization head (NLCDetection)
            pred_mask = self.SegNet(feat)[0]
            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                      align_corners=True)

        # Convert tensor to NumPy array
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

        # Normalize values (0 to 255)
        pred_mask_np = (pred_mask_np * 255).astype(np.uint8)

        # Compute pixelwise confidence score
        confidence = self.confidence_score_pixelwise(pred_mask_np)

        # Convert NumPy array to image
        mask_image = Image.fromarray(pred_mask_np)

        # Convert image to Base64 (alternative method)
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")  # Save as PNG format
        buffered.seek(0)  # Move to the start of the buffer
        base64_mask = base64.encodebytes(buffered.read()).decode("utf-8").replace("\n", "")

        return base64_mask, confidence
