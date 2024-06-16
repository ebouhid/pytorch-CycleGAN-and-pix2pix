"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import numpy as np
from data.image_folder import make_dataset
from torchvision.transforms import ToTensor, Compose
import os
# from PIL import Image

class HaralickDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.image_paths = sorted(make_dataset(self.root, opt.max_dataset_size))  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]   # needs to be a string
        if 'A' in path:
            path_A = path
            path_B = path.replace('A', 'B')
        else:
            path_A = path.replace('B', 'A')
            path_B = path

        # Check if the files exist
        if not os.path.exists(path_A):
            return self.__getitem__(np.random.randint(len(self.image_paths)))
        if not os.path.exists(path_B):
            return self.__getitem__(np.random.randint(len(self.image_paths)))

        data_A = np.load(path_A).astype(np.float32)
        data_B = np.load(path_B).astype(np.float32)

        # Classification - define the label
        label = 0. # forest
        if 'non_forest' in path:
            label = 1.
        elif 'not_analyzed' in path:
            label = 1.

        return {'A': data_A, 'B': data_B, 'A_paths': path_A, 'B_paths': path_B, 'label': label}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
