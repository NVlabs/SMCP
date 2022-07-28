# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from collections import OrderedDict
import math
from typing import Callable, List, Optional, OrderedDict, Tuple

import torch
from torch import nn

from smcp.detection.models.resnet_backbone import BasicBlock, Bottleneck, ResNetBackbone

def get_feat_extractor(arch: str) -> Tuple[nn.Module, int]:
    # Set building block
    if arch in ["resnet18", "resnet34"]:
        block = BasicBlock
    else:
        block = Bottleneck

    # Set number of layers
    if arch == "resnet18":
        layers = [2, 2, 2]
    elif arch in ["resnet34", "resnet50"]:
        layers = [3, 4, 6]
    elif arch in ["resnet101"]:
        layers = [3, 4, 23]
    elif arch == "resnet152":
        layers = [3, 8, 36]
    else:
        raise NotImplementedError

    feat_extractor = ResNetBackbone(block, layers, zero_init_residual=True)

    return feat_extractor, feat_extractor.out_channels

def create_and_init_feature_block(
    label: str, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
    dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, bias: bool = False
) -> OrderedDict[str, nn.Module]:
    layers = []

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size,
        padding=padding, stride=stride, dilation=dilation, bias=bias
    )
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    layers.append((f"conv{label}", conv))

    if norm_layer:
        layers.append((f"bn{label}", norm_layer(out_channels)))

    layers.append((f"relu{label}", nn.ReLU(inplace=True)))

    return layers

def get_ssd_additional_layers(
    in_channels: int, bias: bool = False, batch_norm: bool = False, is_ssd300: bool = True
) -> List[nn.Module]:
    all_layers: List[nn.Module] = []

    norm_layer = nn.BatchNorm2d if batch_norm else None

    # Layers until conv8_2
    layers_1 = nn.Sequential(OrderedDict([
        *create_and_init_feature_block("5-1", in_channels, 512, 3, padding=1, norm_layer=norm_layer, bias=bias),
        *create_and_init_feature_block("5-2", 512, 512, 3, stride=2, padding=1, norm_layer=norm_layer, bias=bias),
    ]))
    all_layers.append(layers_1)

    layers_2 = nn.Sequential(OrderedDict([
        *create_and_init_feature_block("8-1", 512, 256, 1, padding=0, norm_layer=norm_layer, bias=bias),
        *create_and_init_feature_block("8-2", 256, 512, 3, stride=2, padding=1, norm_layer=norm_layer, bias=bias),
    ]))
    all_layers.append(layers_2)

    # Layers until conv9_2
    layers_3 = nn.Sequential(OrderedDict([
        *create_and_init_feature_block("9-1", 512, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
        *create_and_init_feature_block("9-2", 128, 256, 3, stride=2, padding=1, norm_layer=norm_layer, bias=bias),
    ]))
    all_layers.append(layers_3)

    # Layers until conv10_2
    if is_ssd300:
        layers_4 = nn.Sequential(OrderedDict([
            *create_and_init_feature_block("10-1", 256, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
            *create_and_init_feature_block("10-2", 128, 256, 3, padding=0, norm_layer=norm_layer, bias=bias),
        ]))
    else:
        layers_4 = nn.Sequential(OrderedDict([
            *create_and_init_feature_block("10-1", 256, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
            *create_and_init_feature_block("10-2", 128, 256, 3, stride=2, padding=1, norm_layer=norm_layer, bias=bias),
        ]))
    all_layers.append(layers_4)

    # Layers until conv11_2
    if is_ssd300:
        layers_5 = nn.Sequential(OrderedDict([
            *create_and_init_feature_block("11-1", 256, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
            *create_and_init_feature_block("11-2", 128, 256, 3, padding=0, norm_layer=norm_layer, bias=bias),
        ]))
    else:
        layers_5 = nn.Sequential(OrderedDict([
            *create_and_init_feature_block("11-1", 256, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
            *create_and_init_feature_block("11-2", 128, 256, 3, stride=2, padding=1, norm_layer=norm_layer, bias=bias),
        ]))
    all_layers.append(layers_5)

    if not is_ssd300:
        layers_6 = nn.Sequential(OrderedDict([
            *create_and_init_feature_block("12-1", 256, 128, 1, padding=0, norm_layer=norm_layer, bias=bias),
            *create_and_init_feature_block("12-2", 128, 256, 4, padding=1, norm_layer=norm_layer, bias=bias),
        ]))
        all_layers.append(layers_6)
    else:
        all_layers.append(None)

    return all_layers

class SSDClassifierBlock(nn.Module):
    def __init__(self, in_channels: int, num_boxes: int, num_classes: int, kernel_size: int, padding: int = 0):
        super().__init__()

        self.num_classes = num_classes

         # Localization prediction convolution (predict offsets w.r.t prior-boxes)
        self.loc = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=kernel_size, padding=padding)

        # Class prediction convolutios (predict classes in localization boxes)
        self.cl = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=kernel_size, padding=padding)

        # Initialize convolutions" parameters
        for c in self.children():
            nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        loc_out = self.loc(x)
        loc_out = loc_out.permute(0, 2, 3, 1).contiguous()
        loc_out = loc_out.view(batch_size, -1, 4)

        c_out = self.cl(x)
        c_out = c_out.permute(0, 2, 3, 1).contiguous()
        c_out = c_out.view(batch_size, -1, self.num_classes)

        return loc_out, c_out

class SSDPredictions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See "cxcy_to_gcxgcy" in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for "background" = no object.
    """

    def __init__(self, num_classes: int, conv4_out_channels: int, is_ssd300: bool = True):
        """
        :param n_classes: number of different types of objects
        """
        super(SSDPredictions, self).__init__()

        self.is_ssd300 = is_ssd300

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4
        }
        if not self.is_ssd300:
            n_boxes["conv10_2"] = 6
            n_boxes["conv12_2"] = 4
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        self.conv4_3 = SSDClassifierBlock(conv4_out_channels, n_boxes["conv4_3"], num_classes, 3, padding=1)
        self.conv7 = SSDClassifierBlock(512, n_boxes["conv7"], num_classes, 3, padding=1)
        self.conv8_2 = SSDClassifierBlock(512, n_boxes["conv8_2"], num_classes, 3, padding=1)
        self.conv9_2 = SSDClassifierBlock(256, n_boxes["conv9_2"], num_classes, 3, padding=1)
        self.conv10_2 = SSDClassifierBlock(256, n_boxes["conv10_2"], num_classes, 3, padding=1)
        self.conv11_2 = SSDClassifierBlock(256, n_boxes["conv11_2"], num_classes, 3, padding=1)

        if not self.is_ssd300:
            self.conv12_2 = SSDClassifierBlock(256, n_boxes["conv12_2"], num_classes, 3, padding=1)

    def forward(
        self, conv4_3_feats: torch.Tensor, conv7_feats: torch.Tensor, conv8_2_feats: torch.Tensor,
        conv9_2_feats: torch.Tensor, conv10_2_feats: torch.Tensor, conv11_2_feats: torch.Tensor,
        conv12_2_feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38) / ((N, 512, 64, 64) for SSD512)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19) / (N, 1024, 32, 32)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10) / (N, 512, 16, 16)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5) / (N, 256, 8, 8)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3) / (N, 256, 4, 4)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1) / (N, 256, 2, 2)
        :param conv12_2_feats: conv12_2 feature map, a tensor of dimensions (N/A) / (N, 256, 1, 1)
        :return: 8732 or 24564 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Predict localization boxes" bounds (as offsets w.r.t prior-boxes) and predict classes in localization boxes
        l_conv4_3, c_conv4_3 = self.conv4_3(conv4_3_feats) # (N, 5776, *), there are a total 5776 boxes on this feature map
        l_conv7, c_conv7 = self.conv7(conv7_feats) # (N, 2166, *), there are a total 2116 boxes on this feature map
        l_conv8_2, c_conv8_2 = self.conv8_2(conv8_2_feats) # (N, 600, *)
        l_conv9_2, c_conv9_2 = self.conv9_2(conv9_2_feats) # (N, 150, *)
        l_conv10_2, c_conv10_2 = self.conv10_2(conv10_2_feats) # (N, 36, *)
        l_conv11_2, c_conv11_2 = self.conv11_2(conv11_2_feats) # (N, 4, *)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = [l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2]
        class_scores = [c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2]

        if not self.is_ssd300:
            l_conv12_2, c_conv12_2 = self.conv12_2(conv12_2_feats)  # (N, 4, *)

            locs.append(l_conv12_2)
            class_scores.append(c_conv12_2)

        locs = torch.cat(locs, dim=1)  # (N, 8732/24564, 4)
        classes_scores = torch.cat(class_scores, dim=1)  # (N, 8732/24564, n_classes)

        return locs, classes_scores

def create_VOC_prior_boxes(is_ssd300: bool = True) -> torch.Tensor:
    """
    Create the 8732/24564 prior (default) boxes for the SSD300/SSD512, as defined in the paper.

    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4) or (24564, 4)
    """
    if is_ssd300:
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
    else:
        fmap_dims = {'conv4_3': 64,
                     'conv7': 32,
                     'conv8_2': 16,
                     'conv9_2': 8,
                     'conv10_2': 4,
                     'conv11_2': 2,
                     'conv12_2': 1}

        obj_scales = {'conv4_3': 0.07,
                      'conv7': 0.15,
                      'conv8_2': 0.30,
                      'conv9_2': 0.45,
                      'conv10_2': 0.60,
                      'conv11_2': 0.75,
                      'conv12_2': 0.90}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 3., 0.5, .333],
                         'conv11_2': [1., 2., 0.5],
                         'conv12_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.tensor(prior_boxes, dtype=torch.float32)  # (8732, 4) or (24564, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4) or (24564, 4)

    return prior_boxes

class SSD(nn.Module):
    def __init__(
        self, arch: str = "SSD300", backbone: str = "resnet50", num_classes: int = 21,
        bias: bool = False, batch_norm: bool = False
    ):
        super(SSD, self).__init__()
        self.is_ssd300 = (arch == "SSD300")
        self.num_classes = num_classes

        self.f_0, out_channels = get_feat_extractor(backbone)
        self.features_1, self.features_2, self.features_3, self.features_4, self.features_5, self.features_6 = \
            get_ssd_additional_layers(out_channels, bias, batch_norm, self.is_ssd300)

        self.predictor = SSDPredictions(num_classes, out_channels, self.is_ssd300)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats_0 = self.f_0(image)  # (N, 512, 38, 38) or (N, 512, 64, 64)

        feats_1 = self.features_1(feats_0)  # (N, 1024, 19, 19) or (N, 1024, 32, 32)
        feats_2 = self.features_2(feats_1)  # (N, 512, 10, 10) or (N, 512, 16, 16)
        feats_3 = self.features_3(feats_2)  # (N, 256, 5, 5) or (N, 256, 8, 8)
        feats_4 = self.features_4(feats_3)  # (N, 256, 3, 3) or (N, 256, 4, 4)
        feats_5 = self.features_5(feats_4)  # (N, 256, 1, 1) or (N, 256, 2, 2)
        if not self.is_ssd300:
            feats_6 = self.features_6(feats_5)  # (N, 256, 1, 1)
        else:
            feats_6 = None

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.predictor(feats_0, feats_1, feats_2, feats_3, feats_4, feats_5, feats_6)

        return locs, classes_scores  # (N, 8732, 4), (N, 8732, n_classes) or (N, 24564, 4), (N, 24564, n_classes)

def get_ssd_model(arch: str = "SSD300", backbone: str = "resnet50", pretrained: Optional[str] = None, **kwargs) -> SSD:
    model = SSD(arch, backbone, **kwargs)

    if pretrained:
        state_dict = torch.load(pretrained, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

    return model
