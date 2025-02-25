# run.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import save_image
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData

class Pscc:
    def __init__(self, args):
        self.device_ids = [Id for Id in range(torch.cuda.device_count())]
        self.device = torch.device('cuda:0')

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

        self.load_network_weights()

    def load_network_weights(self):
        # load FENet weight
        self.FENet = self.FENet.to(self.device)
        self.FENet = nn.DataParallel(self.FENet, device_ids=self.device_ids)
        self.load_weight(self.FENet, self.FENet_checkpoint_dir, self.FENet_name)

        # load SegNet weight
        self.SegNet = self.SegNet.to(self.device)
        self.SegNet = nn.DataParallel(self.SegNet, device_ids=self.device_ids)
        self.load_weight(self.SegNet, self.SegNet_checkpoint_dir, self.SegNet_name)

        # load ClsNet weight
        self.ClsNet = self.ClsNet.to(self.device)
        self.ClsNet = nn.DataParallel(self.ClsNet, device_ids=self.device_ids)
        self.load_weight(self.ClsNet, self.ClsNet_checkpoint_dir, self.ClsNet_name)

    def load_weight(self, net, checkpoint_dir, name):
        weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
        net_state_dict = torch.load(weight_path, map_location='cuda:0', weights_only=True)
        net.load_state_dict(net_state_dict)
        print('{} weight-loading succeeds'.format(name))

    def predict(self, image: torch.Tensor):
        image = image.to(self.device)

        # backbone network
        self.FENet.eval()
        feat = self.FENet(image)

        # localization head
        self.SegNet.eval()
        pred_mask = self.SegNet(feat)[0]

        pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                  align_corners=True)

        # classification head
        self.ClsNet.eval()
        pred_logit = self.ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'
        return pred_tag
