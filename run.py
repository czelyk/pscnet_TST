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
        self.device = torch.device('cuda:0')
        self.args = args

        # Define networks
        self.FENet = get_seg_model(get_hrnet_cfg()).to(self.device)
        self.SegNet = NLCDetection(self.args).to(self.device)
        self.ClsNet = DetectionHead(self.args).to(self.device)

        # Wrap with DataParallel
        self.FENet = nn.DataParallel(self.FENet, device_ids=self.device_ids)
        self.SegNet = nn.DataParallel(self.SegNet, device_ids=self.device_ids)
        self.ClsNet = nn.DataParallel(self.ClsNet, device_ids=self.device_ids)

        # Load weights
        self.load_weight(self.FENet, './checkpoint/HRNet_checkpoint', 'HRNet')
        self.load_weight(self.SegNet, './checkpoint/NLCDetection_checkpoint', 'NLCDetection')
        self.load_weight(self.ClsNet, './checkpoint/DetectionHead_checkpoint', 'DetectionHead')

    def load_weight(self, net, checkpoint_dir, name):
        weight_path = f'{checkpoint_dir}/{name}.pth'
        net.load_state_dict(torch.load(weight_path, map_location=self.device))
        print(f'{name} weight-loading succeeds')

    def predict(self, image: torch.Tensor):
        image = image.float() / 255.0
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = image.to(self.device)

        # Run models
        with torch.no_grad():
            feat = self.FENet(image)
            pred_mask = self.SegNet(feat)[0]
            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)
            pred_logit = self.ClsNet(feat)

        # Compute forged confidence
        sm = nn.Softmax(dim=1)
        pred_prob = sm(pred_logit)
        forged_confidence = pred_prob[0, 1].item() * 10

        # Convert mask to base64
        pred_mask_np = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(pred_mask_np)
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        base64_mask = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return base64_mask, forged_confidence