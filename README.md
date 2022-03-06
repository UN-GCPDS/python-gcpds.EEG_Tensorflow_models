# python-gcpds.EEG_Tensorflow_models
Deep learning models applied to EEG signals on tensorflow 2.x

## Datasets
|  Name  | Subjects | fs[Hz] | Classes |
| ----------- | ----------- | ----------- | ----------- |
| [BCI2a](http://www.bbci.de/competition/iv/desc_2a.pdf) | 9 | 250 | Left hand / Right hand / Feet / Toungue |
| [GIGA17](http://gigadb.org/dataset/100295) | 52 | 500 | Left hand / Right hand |

## State-of-the-art methods
* [DeepConvNet:](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730) Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization.
* [ShallowConvNet:](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730) Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization.
* [EEGNet:](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta?casa_token=4ODyqJk1R9QAAAAA:UPFDjLMNl6vB0n76FzjquukcdHLCPJUxqZ33jyhcUR2uK5OmyGX8BrHoaXARF3g0G70H6CG7K8o6) A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
* [DMTL_BCI:](https://ieeexplore.ieee.org/abstract/document/8852362) EEG-Based Motor Imagery Classification with Deep Multi-Task Learning.
* [PST-Attention:](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7759669/) Parallel Spatial–Temporal Self-Attention CNN-Based Motor Imagery Classification for BCI.
* [TCNet-Fusion:](https://www.sciencedirect.com/science/article/pii/S1746809421004237?casa_token=XsRO0Coq4m0AAAAA:bWrqLHTs8OyrMbQABAAe4wBw1JOOZQtSYmcBoDl9_qzubD2hOVzKwmSVZkqkNkXmskamLPs4sJo) Electroencephalography-based motor imagery classification using temporal convolutional network fusion.

## Proposed approaches
* [MTVAE:](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Examples/BCI2a/mtvae.ipynb) Multi-task variational autoencoder.
* [ShallowConvNet_1Conv2d: ](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Examples/BCI2a/shallowconvnet_version1conv2d.ipynb) Variation of ShallowConvNet with only a convolutional 2D layer.
* [ShallowConvNet_1Conv2d_rff: ](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Examples/BCI2a/shallownet_1conv2d_rff_conv2d.ipynb): Variation of ShallowConvNet_1Conv2D using Random Fourier Features layer.
* [Deep&Wide Learning using Gradient Class Activation Maps for Enhanced Physiological Interpretability of Motor Imagery Skills](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/tree/main/Experimental/DW_LCAM/)
* [CNN_DW_ITL](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Experimental/CNN_DW_ITL/Cuaderno_Gauss_y_CSP%2BCWT.ipynb)
* [Preprocessing_MI](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/tree/main/Experimental/Preprocessing_MI/)

## Install

1. Clone this repo.

```
git clone https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models
```

2. Install repo.

On personal laptop
```
pip install -e git+https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models.git#egg=EEG_Tensorflow_models
```
On Google Colab
```
pip install -U git+https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models.git
```
## Results

- BCI2a:

  <img src="https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Results/comparing_mean_acc.jpg" width="70%">
  
- GIGA17:

  <img src="https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models/blob/main/Results/comparing_mean_acc_giga.jpg" width="70%">
