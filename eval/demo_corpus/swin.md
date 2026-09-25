# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Abstract
This paper presents a new vision Transformer, called Swin Transformer, that capably
serves as a general-purpose backbone for computer vision. Challenges in adapting
Transformer from language to vision arise from differences between the two domains.

## 3 Method
Swin Transformer builds a hierarchical representation by starting from small-sized
patches and gradually merging neighboring patches in deeper Transformer layers.
The key design element is the shifted windowing scheme: self-attention is computed
within non-overlapping local windows, and in the next layer the window partition is
shifted, allowing cross-window connections.

The computational complexity of the shifted window scheme is linear in image size,
in contrast to the quadratic complexity of standard global self-attention. This makes
Swin suitable for dense prediction tasks such as semantic segmentation and object
detection.

## 4 Experiments
We evaluate Swin Transformer on ImageNet-1k classification, COCO object detection,
and ADE20K semantic segmentation. Pretraining uses ImageNet-21k.

Results: Swin-L achieves 87.3% top-1 accuracy on ImageNet-1k. On COCO test-dev it
reaches 58.7 box AP and 51.1 mask AP. On ADE20K it reaches 53.5 mIoU, surpassing
the previous state of the art by a significant margin.

## 5 Conclusion
Swin Transformer achieves strong performance across a broad range of vision tasks,
demonstrating the potential of Transformer-based models as vision backbones.
