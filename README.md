### This is the PyTorch implementation of paper Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs (https://arxiv.org/pdf/1907.06724.pdf)

This version doesn't have BatchNorm layers for fine-tuning. If you want to use such model for training, you should add these layers manually.

The procedure for conversion was pretty interesting:
  1. I unpacked ARCore iOS framework and took tflite model of facemesh. You can download it [here](https://developers.google.com/ar/develop/ios/augmented-faces/quickstart)
  2. Paper doesn't state any architecture details, so I looked at [Netron](https://github.com/lutzroeder/netron) graph visualization to reverse-engineer number of input-output channels and operations.
  3. Made them in pytorch and transfer raw weights from tflite file semi-manually into pytorch model definition. (see Convert-FaceMesh.ipynb for details)
