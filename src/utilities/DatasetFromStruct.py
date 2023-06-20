import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetFromStruct(Dataset):
    """
    This class reads a Dataset based on the folder structure.

    data_dir
    ├── images
    │   ├── event_001_pre_disaster.tif
    │   ├── event_001_post_disaster.tif
    │   ├── event_002_pre_disaster.tif
    │   ├── event_002_post_disaster.tif
    │   ├── ...
    ├── masks
    │   ├── event_001_pre_disaster.tif
    │   ├── event_001_post_disaster.tif
    │   ├── event_002_pre_disaster.tif
    │   ├── event_002_post_disaster.tif
    │   ├── ...

    Attributes
    ----------
    data_dir : pathlib.Path
        Path object to main level of data
    data_statistics : dict
        Dictionary including the mean and standard deviation to be used for normalizing the data
    normalize : bool
        Whether data shall be normalized
    augmentation : bool
        Whether data shall be augmented
    """

    def __init__(self, data_dir, data_statistics, normalize=True):
        # global path to data directory
        self.data_dir = data_dir
        # set list of global paths to data
        self.img_pre_paths = list(self.data_dir.joinpath('images').glob('*pre*.tif'))
        self.img_post_paths = list(self.data_dir.joinpath('images').glob('*post*.tif'))
        self.cls_paths = list(self.data_dir.joinpath('masks').glob('*post*.tif'))
        # check whether number of images match
        assert len(self.img_pre_paths) == len(self.img_post_paths)
        assert len(self.img_pre_paths) == len(self.cls_paths)
        # mean and std of dataset
        self.data_statistics = data_statistics
        # set boolean for normalization and augmentation
        self.normalize = normalize

    def __len__(self):
        return len(self.img_pre_paths)

    def __getitem__(self, i):

        # get image name
        image_name = self.img_pre_paths[i].stem.replace('_pre_disaster', '')
        # open images as numpy arrays with shape (bands, x, y) for RGB and (x, y) for mask
        # dtype float32 for raster -> FloatTensor as expected by loss function
        pre_img = tifffile.imread(str(self.img_pre_paths[i])).astype(np.float32) / 255.0
        post_img = tifffile.imread(str(self.img_post_paths[i])).astype(np.float32) / 255.0
        # dtype int64 for masks -> LongTensor as expected by loss function
        cls_mask = tifffile.imread(str(self.cls_paths[i])).astype(np.int64)

        # TODO
        # replace non-classified pixels with background
        cls_mask = np.where(cls_mask == 5, 0, cls_mask)

        # transpose RGB to (x, y, bands)
        pre_img = np.transpose(pre_img, [1, 2, 0])
        post_img = np.transpose(post_img, [1, 2, 0])

        # normalize data
        if self.normalize is True:
            # load statistics from dictionary
            pre_mean = self.data_statistics['pre_mean']
            pre_std = self.data_statistics['pre_std']
            post_mean = self.data_statistics['post_mean']
            post_std = self.data_statistics['post_std']
            # normalize data and convert numpy array to tensor with shape (bands, x, y)
            norm_pre = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pre_mean, std=pre_std)
            ])
            norm_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=post_mean, std=post_std)
            ])
            # apply transformation
            pre_img = norm_pre(pre_img)
            post_img = norm_post(post_img)

        # convert numpy array to tensor with (bands, x, y)
        else:
            pre_img = transforms.ToTensor()(pre_img)
            post_img = transforms.ToTensor()(post_img)

        # stack pre- and post-image
        stack_img = torch.cat((pre_img, post_img), dim=0)

        # return images as Pytorch tensors with shape (1, bands, x, y)
        return stack_img, cls_mask
