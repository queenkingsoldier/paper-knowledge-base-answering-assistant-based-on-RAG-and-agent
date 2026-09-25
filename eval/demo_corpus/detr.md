# End-to-End Object Detection with Transformers (DETR)

## Abstract
We present a new method that views object detection as a direct set prediction problem.
Our approach streamlines the detection pipeline, effectively removing the need for many
hand-designed components like a non-maximum suppression procedure or anchor generation.

## 3 Method
DETR combines a CNN backbone with a Transformer encoder-decoder. The backbone extracts
a feature map, which is flattened and augmented with positional encodings before being
passed to the encoder. The decoder uses a fixed set of N learned object queries.

The key insight is the use of bipartite matching between predicted and ground-truth
objects, computed with the Hungarian algorithm. The matching cost is a combination of
classification loss, L1 box distance and generalized IoU loss. Because the matching is
one-to-one, DETR does not need non-maximum suppression at inference time.

## 4 Experiments
DETR is trained on COCO 2017. We use AdamW with an initial learning rate of 1e-4,
weight decay 1e-4, and a batch size of 64. The model is trained for 300 epochs.

Results: DETR achieves 42.0 AP on COCO val, comparable to a well-tuned Faster R-CNN
baseline, but with a much simpler pipeline. DETR excels on large objects but shows
lower performance on small objects.

## 5 Conclusion
DETR offers a new paradigm for object detection: it is conceptually simple and does
not require specialized libraries, yet matches the accuracy of highly optimized
detection frameworks.
