import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from utils.train_utils import *
from natsort import natsorted


class Random_Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-1])
        return tensor


class Random_Vertical_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-2])
        return tensor


class DataSet(torch.utils.data.Dataset):
    def __init__(self, config, eval):
        super(DataSet, self).__init__()
        self.config = config
        self.eval = eval
        self.gt_dir = self.config['train_data_dir'] if not self.eval else \
            self.config['eval_data_dir']
        self.burst_length = self.config['burst_length']
        self.patch_size = self.config['patch_size']
        self.color = self.config['color']

        # data augmentations
        self.vertical_flip = Random_Vertical_Flip(p=0.5)
        self.horizontal_flip = Random_Horizontal_Flip(p=0.5)

        # paths list of ground-truth images
        self.gt_paths = natsorted(
            [os.path.join(self.gt_dir, x) for x in os.listdir(self.gt_dir) if
             is_image(x)])
        # Note that data cleaning is required  to exclude images with sizes
        # smaller than self.patch_size.
        self.gt_paths = exclude_too_small_images(self.gt_paths, self.patch_size)

    @staticmethod
    def crop_random(tensor, patch_size):
        return random_crop(tensor, 1, patch_size)[0]

    # get an item according to the random index
    def __getitem__(self, index):
        gt = Image.open(self.gt_paths[index]).convert(
            'RGB' if self.color else 'L')
        gt = transforms.ToTensor()(gt)

        # data augmentation
        if not self.eval:
            gt = self.crop_random(gt, self.patch_size)
            gt = self.vertical_flip(gt)
            gt = self.horizontal_flip(gt)

        # noisy burst synthesis with AWGN
        img_burst = []
        for i in range(self.burst_length):
            img_burst.append(gt)
        image_burst = torch.stack(img_burst, dim=0)
        # AWGN with sigma 25
        GaussianNoise = np.random.normal(loc=0., scale=25.,
                                         size=image_burst.shape) / 255.
        GaussianNoise = torch.from_numpy(GaussianNoise).type_as(image_burst)
        image_burst = torch.clamp(image_burst + GaussianNoise, 0., 1.)

        return image_burst, gt

    def __len__(self):
        return len(self.gt_paths)
