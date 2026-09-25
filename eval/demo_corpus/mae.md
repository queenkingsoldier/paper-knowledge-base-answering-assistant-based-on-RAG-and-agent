# Masked Autoencoders Are Scalable Vision Learners (MAE)

## Abstract
This paper shows that masked autoencoders (MAE) are scalable self-supervised learners
for computer vision. Our MAE approach is simple: we mask random patches of the input
image and reconstruct the missing pixels.

## 3 Method
MAE is based on two core designs. First, we develop an asymmetric encoder-decoder
architecture. The encoder operates only on the visible subset of patches (without mask
tokens), while a lightweight decoder reconstructs the original image from the latent
representation and mask tokens.

Second, we find that masking a high proportion of the input image — for example 75% —
yields a nontrivial and meaningful self-supervisory task. This high masking ratio
creates a difficult task that requires holistic understanding of the scene rather than
extrapolation from nearby visible patches. The reconstruction target is normalized
pixels within each patch.

Masking a high ratio also substantially reduces compute: the encoder processes only
25% of the patches.

## 4 Experiments
We pretrain MAE on ImageNet-1k and ImageNet-21k. Pretraining uses AdamW with a batch
size of 4096 and 1600 epochs for the ImageNet-1k setting.

Results: MAE pretraining on ImageNet-1k with ViT-Huge reaches 87.8% top-1 fine-tuning
accuracy, improving over supervised pretraining. MAE also transfers well to detection
and segmentation with only a small fine-tuning overhead.

## 5 Conclusion
MAE is a scalable and simple self-supervised approach. It demonstrates that masked
autoencoding, a technique from NLP, can be an effective pretraining task in vision.
