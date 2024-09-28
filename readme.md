# ACNN

## introduction

![pic1](https://github.com/THUMetaOptics/ACNN/blob/main/pic/pic1.png)

** Figure 1 | ** Schematic of the system using a metasurface to perform multi-kernel convolution of the input image and project all features to the sensor. The features are then fed into a digital backend for classification.

The ACNN consists of a metasurface-based convolution layer and a digital backend, as shown in Fig.1. Metasurface performs multi-kernel optical convolution of the input image. After optical convolution, we perform ReLU activation, max pooling, flattening, and dropout operations on the 8 feature maps. The resulting output is then fed into a fully connected layer for classification, as shown in Fig.2.

![pic2](.\pic\pic2.png)

**Figure 2 |  **Network structure with convolution operations done in optics, which takes up most of the FLOPs. A digital backend, including ReLU activation, max pooling, flattening, dropout, and fully connected layer, performs classification after the optical convolution.

We develop an end-to-end frequency domain training method (Fig. 3) to build convolution kernels with arbitrary shapes and large receptive fields.

![pic3](.\pic\pic3.png)

**Figure 3 |  **Training in the spatial frequency domain involves performing a Fast Fourier Transform (FFT) on the object to obtain its frequency domain distribution, which is multiplied by the OTF derived from the Normalized Autocorrelation Function (Acorr) of the pupil phase distribution. An Inverse Fast Fourier Transform (IFFT) is then performed to generate the final image, which is flattened and fed into the digital backend for classification. The black and orange arrows represent the forward model and the backpropagation-based end-to-end training process, respectively.

## training and test

ACNN_no_parallel_acceleration.ipynb includes training and testing process, which 16 optical kernels are calculated successively. This method performs better on Fashion-MNIST and CIFAR-10 datasets.

ACNN_with_parallel_acceleration.ipynb includes training and testing process, which 16 optical kernels are calculated synchronously. This method is much quick. Its mainly used to train on the MNIST dataset.

## Acceleration of Autocorrelation Function 

We have written the `Acorr()`  function in the program to implement Autocorrelation Function which is a convolution of self and self-conjugation. However, this convolution method results in long training times. Based on the theory that spatial convolution is equivalent to frequency domain multiplication, we have written `Acorr_FFT()` function to accelerate the process.

