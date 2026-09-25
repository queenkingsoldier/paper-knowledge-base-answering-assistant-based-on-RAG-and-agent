# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## Abstract
We show that a pure transformer applied directly to sequences of image patches can
perform very well on image classification tasks. We propose the Vision Transformer (ViT),
which treats an image as a sequence of flattened 2D patches.

## 3 Method
We split an image into fixed-size patches of 16x16 pixels, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard
Transformer encoder. An extra learnable "classification token" is prepended to the
sequence, and its final hidden state serves as the image representation.

The Transformer encoder consists of alternating layers of multiheaded self-attention
(MSA) and MLP blocks. Layernorm (LN) is applied before every block, and residual
connections after every block.

## 4 Experiments
We pretrain ViT on the ImageNet-21k dataset (14 million images, 21843 classes) and
the JFT-300M dataset (303 million images). We then fine-tune on downstream tasks.
For fine-tuning we use SGD with momentum, batch size 512, and a learning rate of 0.03.

Results: ViT-H/14 reaches 88.55% top-1 accuracy on ImageNet when pretrained on
JFT-300M. Notably, ViT requires substantially less compute to train than comparable
convolutional networks, but it does not generalize well when trained on insufficient
amounts of data — ViT trained on ImageNet-1k alone underperforms ResNet.

## 5 Conclusion
We introduced the Vision Transformer. While transformers have become the de-facto
standard in NLP, their application to computer vision remained limited. We show that
this reliance on CNNs is not necessary.
