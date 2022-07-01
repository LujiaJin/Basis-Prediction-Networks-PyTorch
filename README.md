# Basis-Prediction-Networks-PyTorch
Reimplementation of "[Basis Prediction Networks for Effective Burst Denoising with Large Kernels](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Basis_Prediction_Networks_for_Effective_Burst_Denoising_With_Large_Kernels_CVPR_2020_paper.pdf)" by using PyTorch.

The source code of the paper was implemented in TensorFlow 1, but not disclosed by the authors due to patent issues. A re-implementation in TensorFlow 2 was shared by Zhihao Xia, the first author of the paper, at [https://github.com/likesum/bpn](https://github.com/likesum/bpn).

To ensure reproducible fidelity, this re-implementation of PyTorch version is based entirely on what is provided in the paper and its supplementary materials, and does not refer to the source code of the TensorFlow 2 version mentioned above. This re-implementation achieves results comparable to those shown in the paper.

The partial work is following [https://github.com/z-bingo/kernel-prediction-networks-PyTorch](https://github.com/z-bingo/kernel-prediction-networks-PyTorch).

## Dependencies
numpy~=1.21.4\
torch~=1.11.0.dev20211210+cu111\
scikit-image~=0.19.2\
imagesize~=1.3.0\
configobj~=5.0.6\
torchvision~=0.12.0.dev20211210+cu111\
Pillow~=8.4.0\
natsort~=8.0.1\
tensorboardX~=2.4.1

## Datasets
Because of the huge amount of data in [Open Images Dateset](https://github.com/cvdfoundation/open-images-dataset), re-implementing the experiments based on this dataset in the paper will bring huge unnecessary time and Energy consumption. Therefore, a simpler experiment to demonstrate is designed.

Additive white Gaussian noise (AWGN) with $\sigma$=25 is added to the [ImageNet 2012 Validation](https://image-net.org/download-images) dataset containing 50,000 images to construct the training set. The same nosie addition process is implemented on [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz), [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/) and [SET14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) to build testing sets. The addition of AWGN is done automatically after loading the data, see [data_provider.py](data_provider.py).

The images in the [ImageNet 2012 Validation](https://image-net.org/download-images) dataset need to be downloaded to the `data/train/ILSVRC2012-Val/` folder. The images in the [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz), [Kodak](http://r0k.us/graphics/kodak/) and [SET14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) datasets need to be downloaded to the `data/test/BSD300/`, `data/test/KODAK/` and `data/test/SET14/`, respectively. Images in [SET14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) dataset has been provided for demonstration.

## Train
### Train from scratch
To train the BPN from scratch on [ImageNet 2012 Validation](https://image-net.org/download-images) dataset with AWGN:

```
python train_and_eval.py --config_file configs/AWGN_RGB.conf -c -m
```

### Continue training from an existing checkpoint
Due to redeployment of code runs or accidental training interruptions, continue training on a previously saved checkpoint:

```
python train_and_eval.py --config_file configs/AWGN_RGB.conf -c -m -ckpt <previously saved checkpoint, best or step number>
```

### Train on grayscale images
If you want to train a BPN model for noise reduction on grayscale images, run:

```
python train_and_eval.py --config_file configs/AWGN_gray.conf -c -m
```

Whether the "color" entry in the config file is "True" or "False" determines that `data_provider.py` will read the image data in RGB or grayscale.

### Train with your own data
Convert your own data into any format that Pillow can read (eg JPEG, PNG, TIF, etc.) and organize it into a specific folder. Write a config file according to your data characteristics and experimental needs and place it in the `configs/` directory. run:

```
python train_and_eval.py --config_file configs/<your_config>.conf -c -m
```

## Test
### Download pre-trained models
My pre-trained models on [ImageNet 2012 Validation](https://image-net.org/download-images) dataset with AWGN of $\sigma$=25 can be found [here](https://drive.google.com/drive/folders/1MJmBC10Y6OTFd_Vrr24ceQaYjik69hO3?usp=sharing), including denoising models for both grayscale and color images.

The downloaded models for grayscale and color images should be placed in the `models/checkpoints/AWGN_gray/` and `models/checkpoints/AWGN_RGB/` directories respectively.

### Test with pre-trained models
To test multi-frame denoising on color images with pre-trained BPN model, run:

```
python train_and_eval.py --config_file configs/AWGN_RGB.conf -c -m --eval -ckpt 75020
```

Similarly, to test multi-frame denoising on grayscale images with pre-trained BPN model, run:

```
python train_and_eval.py --config_file configs/AWGN_gray.conf -c -m --eval -ckpt 87260
```

## Results
### Qualitative evaluation results
Several representative image examples of the denoising results are provided below, and more result images can be found in `results/`.

#### for color images:
<table>
<tr>
<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/blob/main/results/AWGN_RGB/SET14/0_gt.png"/ width="300"> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_RGB/SET14/0_noisy_20.3030dB_0.6210_0.0966_0.9284.png"/ width="300" height=width> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_RGB/SET14/0_pred_28.0294dB_0.8741_0.0397_0.9889"/ width="300" height=width> </center> </td>
</tr>

<tr>
<td><center> Ground Truth </center></td>
<td><center> Noisy (20.30dB) </center></td>
<td><center> Denoised (28.03dB) </center></td>
</tr>

<tr>
<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_RGB/SET14/4_gt.png"/ width="300"> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_RGB/SET14/4_noisy_20.4918dB_0.6354_0.0945_0.9353.png"/ width="300" height=width> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_RGB/SET14/4_pred_32.1668dB_0.9619_0.0246_0.9957.png"/ width="300" height=width> </center> </td>
</tr>

<tr>
<td><center> Ground Truth </center></td>
<td><center> Noisy (20.49dB) </center></td>
<td><center> Denoised (32.17dB) </center></td>
</tr>
</table>

#### for grayscale images:

<table>
<tr>
<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/8_gt.png"/ width="300"> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/8_noisy_18.5804dB_0.3202_0.1178_0.8884"/ width="300" height=width> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/8_pred_27.8295dB_0.8423_0.0406_0.9953"/ width="300" height=width> </center> </td>
</tr>

<tr>
<td><center> Ground Truth </center></td>
<td><center> Noisy (18.58dB) </center></td>
<td><center> Denoised (27.83dB) </center></td>
</tr>

<tr>
<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/11_gt.png"/ width="300"> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/11_noisy_18.8532dB_0.3041_0.1141_0.9082"/ width="300" height=width> </center> </td>

<td> <center> <img src="https://github.com/LujiaJin/Basis-Prediction-Networks-PyTorch/results/AWGN_gray/SET14/11_pred_32.6694dB_0.8572_0.0233_0.9960"/ width="300" height=width> </center> </td>
</tr>

<tr>
<td><center> Ground Truth </center></td>
<td><center> Noisy (18.85dB) </center></td>
<td><center> Denoised (32.67dB) </center></td>
</tr>
</table>

### Quantitative evaluation results
The quantitative evaluation results on the three test sets are also given as follows. In addition to the PSNR selected in the original paper, this re-implementation adds three additional quantitative image quality evaluation metrics, SSIM, RMSE, and Pearson R.

#### for color images:
|          | PSNR (dB) | SSIM  | RMSE  | R     |
| -------- | --------- | ----- | ----- | ----- |
| BSD300   | 33.76     | 0.943 | 0.021 | 0.996 |
| KODAK    | 35.07     | 0.941 | 0.018 | 0.996 |
| SET14    | 33.21     | 0.924 | 0.023 | 0.996 |
| average  | 33.84     | 0.942 | 0.021 | 0.996 |

#### for grayscale images:
|          | PSNR (dB) | SSIM  | RMSE  | R     |
| -------- | --------- | ----- | ----- | ----- |
| BSD300   | 31.43     | 0.918 | 0.028 | 0.992 |
| KODAK    | 33.76     | 0.917 | 0.021 | 0.993 |
| SET14    | 32.24     | 0.909 | 0.026 | 0.995 |
| average  | 31.63     | 0.917 | 0.028 | 0.993 |

## License
All the materials, including the codes and pretrained models, are made freely available for non-commercial use under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/).

#### If you like this repo, Star or Fork to support my work. Thank you!
