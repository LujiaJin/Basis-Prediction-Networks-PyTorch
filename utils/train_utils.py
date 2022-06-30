import numpy as np
import glob
import torch
import shutil
import os
import skimage
import random
from configobj import ConfigObj
from validate import Validator
import imagesize


class MovingAverage(object):
    def __init__(self, n):
        self.n = n
        self._cache = []
        self.mean = 0

    def update(self, val):
        self._cache.append(val)
        if len(self._cache) > self.n:
            del self._cache[0]
        self.mean = sum(self._cache) / len(self._cache)

    def get_value(self):
        return self.mean


def save_checkpoint(state, is_best, checkpoint_dir, n_iter, max_keep=10):
    filename = os.path.join(checkpoint_dir, "{:07d}.pth.tar".format(n_iter))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_dir,
                                     'model_best.pth.tar'))
    files = sorted(os.listdir(checkpoint_dir))
    rm_files = files[0:max(0, len(files) - max_keep)]
    for f in rm_files:
        os.remove(os.path.join(checkpoint_dir, f))


def _represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_checkpoint(checkpoint_dir, best_or_latest='best'):
    if best_or_latest == 'best':
        checkpoint_file = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    elif best_or_latest.isdigit():
        best_or_latest = int(best_or_latest)
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:07d}.pth.tar'.format(best_or_latest))
        if not os.path.exists(checkpoint_file):
            files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
            basenames = [os.path.basename(f).split('.')[0] for f in files]
            iters = sorted([int(b) for b in basenames if _represent_int(b)])
            raise ValueError('Available iterations are ({} requested): {}'.format(best_or_latest, iters))
    else:
        files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
        basenames = [os.path.basename(f).split('.')[0] for f in files]
        iters = sorted([int(b) for b in basenames if _represent_int(b)])
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:07d}.pth.tar'.format(iters[-1]))
    return torch.load(checkpoint_file)


def read_config(config_file, config_spec):
    configspec = ConfigObj(config_spec, raise_errors=True)
    config = ConfigObj(config_file,
                       configspec=configspec,
                       raise_errors=True,
                       file_error=True)
    config.validate(Validator())
    return config


def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()


def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += skimage.metrics.peak_signal_noise_ratio(normalize(target_tf[im_idx, ...]),
                                             normalize(output_tf[im_idx, ...]),
                                             data_range=1.)
        n += 1.0
    return psnr / n


def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += skimage.metrics.structural_similarity(normalize(target_tf[im_idx, ...]),
                                             normalize(output_tf[im_idx, ...]),
                                             channel_axis=2,
                                             K1=0.01, K2=0.03, sigma=1.5,
                                             data_range=1.)
        n += 1.0
    return ssim / n


def calculate_rmse(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    rmse = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        rmse += np.sqrt(skimage.metrics.mean_squared_error(normalize(target_tf[im_idx, ...]),
                                             normalize(output_tf[im_idx, ...])))
        n += 1.0
    return rmse / n


def calculate_pearsonr(output_img, target_img):
    target_tf = torch.clamp(target_img, 0.0, 1.0).cpu().data.numpy()
    output_tf = torch.clamp(output_img, 0.0, 1.0).cpu().data.numpy()
    pearsonr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        pearsonr += np.corrcoef(target_tf[im_idx, ...], output_tf[im_idx, ...])[0, 1]
        n += 1.0
    return pearsonr / n


def normalize(Img):
    if Img.max() > Img.min():
        return (Img - Img.min()) / (Img.max() - Img.min())
    else:
        return Img


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def is_image(filename):
    return any(filename.endswith(extension) for extension in
               [".png", ".jpg", ".jpeg", ".JPEG", ".tif", ".bmp", ".npy"])


def random_crop(im, num_patches, w, h=None):
    h = w if h is None else h
    nw = im.size(-1) - w
    nh = im.size(-2) - h
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is too small {} for the desired size {}". \
                                format((im.size(-1), im.size(-2)), (w, h))
                          )

    idx_w = np.random.choice(nw + 1, size=num_patches)
    idx_h = np.random.choice(nh + 1, size=num_patches)

    result = []
    for i in range(num_patches):
        result.append(im[...,
                         idx_h[i]:(idx_h[i]+h),
                         idx_w[i]:(idx_w[i]+w)])
    return result


def exclude_too_small_images(img_path_list, threshold):
    excluded_path_list = []
    for img_path in img_path_list:
        width, height = imagesize.get(img_path)
        if width >= threshold and height >= threshold:
            excluded_path_list.append(img_path)
    return excluded_path_list
