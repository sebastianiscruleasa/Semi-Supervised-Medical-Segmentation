import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torch.nn.functional import interpolate
from torchvision import transforms


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/volumes/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class SynapseResize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # image, label = torch.from_numpy(sample["image"]), torch.from_numpy(sample["label"])
        # image = image.resize_(self.output_size)
        # label = label.resize_(self.output_size)
        # image = np.array(image)
        # label = np.array(label)

        image, label = sample["image"], sample["label"]
        image = np.resize(image, (224, 224))
        label = np.resize(label, (224, 224))

        # to_pil_transform = transforms.ToPILImage()
        # image = to_pil_transform(image)
        # label = to_pil_transform(label)
        # resize_transform = transforms.Resize(self.output_size)
        # image = resize_transform(image)
        # label = resize_transform(label)
        # image = np.array(image)
        # label = np.array(label)
        sample = {"image": image, "label": label}
        return sample


class SynapseValResize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # image, label = torch.from_numpy(sample["image"]), torch.from_numpy(sample["label"])
        # image = image.view(image.shape[0], 1, image.shape[1], image.shape[2])
        # label = label.view(label.shape[0], 1, label.shape[1], label.shape[2])
        # image = interpolate(image, self.output_size, mode="nearest")
        # label = interpolate(label, self.output_size, mode="nearest")
        # image = image.view(image.shape[0], image.shape[2], image.shape[3])
        # label = label.view(label.shape[0], label.shape[2], label.shape[3])

        # image, label = sample["image"], sample["label"]
        # image = np.resize(image, (89, 224, 224))
        # label = np.resize(label, (89, 224, 224))
        image, label = sample["image"], sample["label"]
        # newImageArray = np.resize(image, (89, 224, 224))
        # newLabelArray = np.resize(label, (89, 224, 224))
        newImageArray = np.empty((image.shape[0], 224, 224))
        newLabelArray = np.empty((image.shape[0], 224, 224))
        # resize_transform = transforms.Resize(self.output_size)
        # to_pil_transform = transforms.ToPILImage()
        for i in range(image.shape[0]-1):
            newImageArray[i, :, :] = np.resize(image[i, :, :], (224, 224))
            newLabelArray[i, :, :] = np.resize(label[i, :, :], (224, 224))

        sample = {"image": newImageArray, "label": newLabelArray}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
