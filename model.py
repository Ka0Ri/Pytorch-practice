import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
                                            deeplabv3_resnet50, deeplabv3_resnet101

from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights, \
                                            DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights \

MODEL = {
    "fcn-m": fcn_resnet50, "fcn-l": fcn_resnet101,
    "deeplab-m": deeplabv3_resnet50, "deeplab-l": deeplabv3_resnet101,
}

WEIGHTS = {
    "fcn-m": FCN_ResNet50_Weights, "fcn-l": FCN_ResNet101_Weights,
    "deeplab-m": DeepLabV3_ResNet50_Weights, "deeplab-l": DeepLabV3_ResNet101_Weights
}

def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        nn.ReLU(inplace=True),
    )

def deconvrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=1),
        nn.ReLU(inplace=True),
    )

class ConvUNet(nn.Module):
    """
    Convolutioal UNet module
    
    """
    def __init__(self, model_configs):
        super(ConvUNet, self).__init__()

        n_filters = model_configs["n_filters"]
        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']
        n_cls = model_configs['n_cls']


        self.downsamling_layers = nn.ModuleList([nn.Sequential(convrelu(model_configs["channel"], n_filters, conv["k"], 1, conv["p"]), 
                                                # nn.MaxPool2d(kernel_size=2)
                                                )])
        for i in range(self.n_layer - 1):
            self.downsamling_layers.append(nn.Sequential(
                # nn.MaxPool2d(kernel_size=2),
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"]),
                )
            )

        transpose_conv = model_configs['transpose_conv']
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_layer - 1, 1, -1):
            self.upsampling_layers.append(deconvrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
            self.upsampling_layers.append(convrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), conv["k"], 1, conv["p"]))

        # Expansive Path
        self.upsampling_layers.append(deconvrelu(2 * n_filters, n_filters, transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
        self.upsampling_layers.append(nn.Conv2d(2 * n_filters, n_cls, conv["k"], padding='same'))
        # self.out = nn.Conv2d(n_filters, model_configs["channel"], conv["k"], padding='same')
       

    def forward(self, x, y=None):

        encode = [x]
        for i in range(self.n_layer):
            down = self.downsamling_layers[i](encode[-1])
            encode.append(down)

        up = encode[-1]
       
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            cat = torch.cat([up, encode[-(i//2) - 2]], dim=1)
            up = self.upsampling_layers[i+1](cat)
            
        return up
    
class WrappingSegment(nn.Module):
    '''
    Wrapping Segmentation model with customized head, 
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 

    - Supported models: fcn, deeplab [-option m: medium, l: large]
    Authors: dtvu1707@gmail.com
    '''
    def __init__(self, model_configs: dict) -> None:

        super(WrappingSegment, self).__init__()
        # Parse parameters
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.n_cls = model_configs['n_cls']
        
        self._model_selection()

    def _model_selection(self):

        # Load pretrained model
        name = self.backbone_name
        assert name in MODEL, "Model %s not found" % name

        if self.is_pretrained:
            base_model = MODEL[name](weights=WEIGHTS[name].COCO_WITH_VOC_LABELS_V1)
        else:
            base_model = MODEL[name](weights=None)
        self.preprocess = WEIGHTS[name].COCO_WITH_VOC_LABELS_V1.transforms()
        self.meta = WEIGHTS[name].COCO_WITH_VOC_LABELS_V1.meta
        
        # turn off gradient
        if self.is_freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        if not self.is_full:
            # Modify last layer
            if any([x in name for x in ["fcn", "deeplab"]]):
                base_model.aux_classifier = None
                num_ftrs = base_model.classifier[-1].in_channels
                base_model.classifier[-1] = nn.Conv2d(num_ftrs, self.n_cls, 1)

        self.base_model = base_model
       
    def forward(self, x, y=None):

        a = self.base_model(x)['out']
        return a