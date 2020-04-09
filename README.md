### This is the PyTorch implementation of paper Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs (https://arxiv.org/pdf/1907.06724.pdf)
![](https://github.com/tensorflow/tfjs-models/blob/master/facemesh/demo.gif?raw=true)

This version doesn't have BatchNorm layers for fine-tuning. If you want to use such model for training, you should add these layers manually.

The procedure for conversion was pretty interesting:
  1. I unpacked ARCore iOS framework and took tflite model of facemesh. You can download it [here](https://developers.google.com/ar/develop/ios/augmented-faces/quickstart)
  2. Paper doesn't state any architecture details, so I looked at [Netron](https://github.com/lutzroeder/netron) graph visualization to reverse-engineer number of input-output channels and operations.
  3. Made them in pytorch and transfer raw weights from tflite file semi-manually into pytorch model definition. (see Convert-FaceMesh.ipynb for details)


#### Input for the model is expected to be cropped face with 25% margin at every side, resized to 192x192 and normalized from -1 to 1
However, `predict_on_image` function normalizes your image itself, so you can even treat resized image as np.array as input

See Inference-FaceMesh.ipynb notebook for usage example 
