# Dynamic Auto-Encoders

## Introduction

One of the (many) problems with auto-encoders is the tight constraint of the bottleneck used to compress the input information. If the bottleneck is too small, the model will not be able to learn anything and generate blurry images.
If the bottleneck is too large, the model will just learn to cheat and memorize the input.

A typical solution to this problem is to do a hyperparameter search to find the best bottleneck size, which can be quite costly.

Here I propose an alternative solution, which is to use a dynamic auto-encoder.
The latent space is masked randomly with zeros, and the decoder is trained to reconstruct the input.
The mask consists of keeping only the first $n$'s dimensions of the latent space.
This is similar to what is done for Principal Component Analysis (PCA), where we can reconstruct the input from the first $n$'s dimensions of the latent space.


## Models

- [ ] Dynamic Auto-Encoder (DyAE)
- [ ] Dynamic Variational Auto-Encoder (DyVAE)
- [ ] Dynamic Vector-Quantized Auto-Encoder (DyVQ-AE)