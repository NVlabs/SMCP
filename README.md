# Soft Masking for Cost-Constrained Channel Pruning

[Soft Masking for Cost-Constrained Channel Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710640.pdf).<br>
Ryan Humble, Maying Shen, Jorge Albericio Latorre, Eric Darve, and Jose M. Alvarez.<br>
ECCV 2022.

Official Pytorch code repository for the "Soft Masking for Cost-Constrained Channel Pruning" paper presented at ECCV 2022 (contact `josea@nvidia.com` for further inquiries).


## Abstract
Structured channel pruning has been shown to significantly accelerate inference time for convolution neural networks (CNNs) on modern hardware, with a relatively minor loss of network accuracy. Recent works permanently zero these channels during training, which we observe to significantly hamper final accuracy, particularly as the fraction of the network being pruned increases. We propose Soft Masking for cost-constrained Channel Pruning (SMCP) to allow pruned channels to adaptively return to the network while simultaneously pruning towards a target cost constraint. By adding a soft mask re-parameterization of the weights and channel pruning from the perspective of removing input channels, we allow gradient updates to previously pruned channels and the opportunity for the channels to later return to the network. We then formulate input channel pruning as a global resource allocation problem. Our method outperforms prior works on both the ImageNet classification and PASCAL VOC detection datasets.

## Training Notes

### NHWC Memory Layout
The code sets the memory layout as NHWC (PyTorch's `channel_last` as described [here](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)). This comes with performance benefits as described in the [NVIDIA DL performance documentation](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).

### Soft channel
We adopt a input channel pruning approach, as described in the paper. The importance and the masks are always done along input channels. However, the cost can be done more flexibly, with the `channel-doublesided-weight` argument: `1` (the default) is to measure with output channels fixed, `0` is to measure with input channels fixed (like HALP), and numbers in between are a combination.

Soft channel pruning only supports limited architectures. We automatically detect the channel structure of the network (which layers need to be pruned together, which layers can be layer pruned, etc.); this detection logic is only known to work for standard ResNet architectures, MobileNetV1, and SSD512-RN50.
Main limitations:
- Group convolutions: Group convolutions are hard to handle, so we only support normal convolutions `groups=1` or depthwise convolutions `groups=in_channels=out_channels`.
- Non-convolution operations: The code only handles convolution, linear, and batch normalization layers as meaningfully interacting with the channels in the network. All other operations are assumed to not change the number of channels nor which dimension of the tensors correpond to the channel (i.e., second dimension for feature maps as the first is the batch dimension). See the further description of the code for more details.

### Getting pruned model and measuring latency
Once training is complete, the slimmed model can be obtained by using the method in `model_clean.py` (which uses `channel_slimmer.py` internally). This removes the pruned channels and saves the network in its entirety (instead of storing as the state dict; see [this](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model) for more details). The code does not support saving/loading just the slimmed state dict.

For measuring latency, we can just load the cleaned model back up and measure the forward pass as usual.

### Training setup
This repository uses [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) to handle most of the training intricacies, including (but not limited to):
- GPU and multi-gpu (via DDP) training
- Sync batch norm (for object detection code)
- Automatic mixed precision
- Logging
- Model checkpoints
- Learning rate schedules
- Metric calculation (Accuracy for classification and MAP for detection)

PyTorch Lightning exposes a nice callback mechanism to integrate custom behavior. We implement a `DynamicPruning` callback class that integrates our pruning code (which does not depend on PyTorch Lightning) and the PyTorch Lightning training setup.

## Experiments

### Image Classification on ImageNet/CIFAR10

Code located in folder [Classification](https://github.com/NVlabs/SMCP/tree/main/smcp/classification)

Run ResNet50 on ImageNet without pruning:
```
 python -m scmp.classification.image_classifier --dataset Imagenet --data-root=/some/path --gpus=1 --fp16
```

With dynamic input channel pruning
```
 ... --prune --channel-type=Global --channel-ratio=0.3
```

See full set of command line arguments [here](https://github.com/NVlabs/SMCP/tree/main/smcp/classification/image_classifier.py).


### Object Detection on Pascal VOC

Code located in folder [Object Detection](https://github.com/NVlabs/SMCP/tree/main/smcp/classification/detection)

Run SSD512-RN50 on PascalVOC without pruning:
```
 python -m smcp.detection.object_detection --dataset PascalVOC --data-root=/som/path --gpus=1 --fp16
```

With dynamic input channel pruning
```
 ... --prune --channel-type=Global --channel-ratio=0.3
```

## Code layout and description (as of 7/19/22)

### Classification
- `image_classifier.py`: main training script for CIFAR10/100/ImageNet
- `image_inference.py`: experimental model cleaning and inference timing script for image classifier models
- `datasets`: folder for CIFAR10/100/ImageNet image classification datasets, written as PyTorch Lightning's `LightningDataModule`s (see their documentation for details)
- `models`: folder for ResNet and MobileNetV1 model definitions

### Detection
- `object_detection.py`: main training script for Pascal VOC
- `metrics.py`: code to take SSD output, convert to detections, and calculate mAP. Includes custom `SSDDetectionMAP` metric to calculate and log mAP periodically during training.
- `datasets`: folder for Pascal VOC object detection dataset, written as PyTorch Lightning's `LightningDataModule` (see their documentation for details)
- `models`: folder for SSD object detection model definitions

### Spare operations
- `base_pruning.py`: base class `BasePruningMethod` for different pruning methods
- `bn_scaling.py`: re-parameterize the BN weights to perform scaling on them
- `channel_costing.py`: different costing functions for cost-constrained pruning
- `channel_pruning.py`: HUGE file for all of dynamic input channel pruning
- `channel_slimmer.py`: automatic slimming code for channel pruning
    - Heavily relies on `torch.fx`
    - Works for both output and input channel pruning
    - See restrictions on supported architectures mentioned above
- `channel_structure.py`: automatic network structure discovery
    - Heavily relies on `torch.fx`
    - Distinguishes between channel acting and channel producing nodes
    - Channel producing: creates any number of output channels *independent* of the number of input channels
    - Channel acting: number of output channels is a function of the number of input channels AND the layer is stateful
- `decay.py`: pruned decay (as defined in this [paper](https://arxiv.org/pdf/2102.04010.pdf))
- `dynamic_pruning.py`: connector between any `BasePruningMethod` and the PyTorch lightning training framework
- `group_knapsack.py`: different solver variants for the multiple-choice knapsack problem described in the paper
- `importance_accumulator.py`: different ways to accumulate importance over the steps between pruning iterations
- `importance.py`: different ways of calculating importance
- `model_clean.py`: wrapper for how to clean/slim the model
- `parameter_masking.py`: re-parameterize the network weights to perform masking; defines several types of masking
- `result.py`: base classes for logging the pruning results (# parameters pruned, unpruned, etc.)
- `scheduler.py`: different pruning schedulers
- `shape_propagation.py`: `torch.fx.Interpreter` to propagate and store feature map shapes through the network

## License
Please check the LICENSE file. SMCP may be used non-commercially. For business inquiries, please contact researchinquiries@nvidia.com.

## Citation
```BibTeX
@article{chang2021image,
  title={Soft Masking for Cost-Constrained Channel Pruning},
  author={Humble, Ryan and Shen, Maying  and Albericio-Latorre, Jorge and Darve, Eric and Alvarez, Jose M},
  journal={ECCV},
  year={2022}
}
```
