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
from PIL import Image
import io
import base64

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda:0')

def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cuda:0', weights_only=True)
    net.load_state_dict(net_state_dict)
    print('{} weight-loading succeeds'.format(name))

def image_to_base64(image_tensor):
    """Converts a tensor to a base64 encoded string."""
    image_tensor = image_tensor.squeeze().cpu().detach().numpy()
    image_tensor = (image_tensor * 255).astype('uint8')  # Normalize values
    image = Image.fromarray(image_tensor)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def test(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)

    # load FENet weight
    FENet = FENet.to(device)
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    # Assume 'image_path' is the single image you want to test
    image_path = args.image_path  # Update this in the args, or pass directly as an argument
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Convert image to tensor and add batch dimension

    with torch.no_grad():
        # backbone network
        FENet.eval()
        feat = FENet(image)

        # localization head
        SegNet.eval()
        pred_mask = SegNet(feat)[0]

        pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=True)

        # classification head
        ClsNet.eval()
        pred_logit = ClsNet(feat)

    # ce
    sm = nn.Softmax(dim=1)
    pred_logit = sm(pred_logit)
    _, binary_cls = torch.max(pred_logit, 1)

    pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

    # Convert pred_mask to base64
    pred_mask_base64 = image_to_base64(pred_mask)

    print(f'The image {image_path} is {pred_tag}')
    print(f'Predicted Mask (Base64): {pred_mask_base64[:100]}...')  # print the first 100 chars of the base64 for brevity

if __name__ == '__main__':
    args = get_pscc_args()
    # Add an argument to pass the image path (if not in args already)
    args.image_path = 'path_to_your_image.jpg'  # You can also get this from command line args
    test(args)
