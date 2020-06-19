---
title: "Bi-linear CNN models for fine-grained visual recognition"
categories:
  - Computer_Vision
tags:
  - Fine-grained Visual recognition
  - CNN arch
---
## Key point
Bilinear CNN utilizes the orderless descriptors which are more powerful to solve the texture classification than the orderful descriptors which store the additional spatial information.

## Summary
Generally, CNN architecture for image classification is composed of convolutional layer and fully connected (FC) layer. At first, conv layer extracts the feature maps (CxHxW shape) from an image, where C is # channels, and H and W are the height and width of the extracted feature maps, respectively. After this, these feature maps are unrolled into 1-D vector, and then they are input to the FC layers; for example, 512x7x7 feature maps are flattened into 25088 neurons and mapped into 4096 feature vectors by FC layer.

The important point is that each neuron from feature map corresponds to the region of the image. Therefore, each neuron can be considered as the descriptor unit for the certain region of the image due to the subsampling process. Since each bit/unit stores spatial information additionally, it can be inferred that less fine-grained information are stored in the orderful descriptors due to the trade-off mechanism. On the other hand, in the fine-grained visual recognition (FGVR) task, the spatial information is not much required but more texture information is valuable. Fisher vector (FV), VLAD and O2P are the orderless descriptors and they outperform FC in FGVR task. Bi-linear CNN model is proposed to train in end-to-end manner to learn the orderless descriptors.

The below diagram shows the Bi-linear CNN architecture briefly. It contains two CNNs that extract two feature maps, F_A and F_B. Then, F_A and F_B are reshaped into the shape of C*M and C*N matrices, respectively. The outer product of these two results in CxMxN matrices which are sum up in axis=0 to give a single MxN matrix. After these steps, MxN matrix is flattened into 1-D vector and input to the softmax layer.

<img src="/assets/imgs/TY_Lin(2015)/arch.PNG" alt="Bi-linear CNN architecture">

Later, I will add more detailed information, such as the backpropagation for outer product of matrices.

## References
<a href="https://medium.com/@ahmdtaha/bilinear-cnn-models-for-fine-grained-visual-recognition-b25ba24d3147">https://medium.com/@ahmdtaha/bilinear-cnn-models-for-fine-grained-visual-recognition-b25ba24d3147</a>

<a href="http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf">http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf</a>