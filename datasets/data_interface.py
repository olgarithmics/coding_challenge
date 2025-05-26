import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataInterface(pl.LightningDataModule):
    """Generic PyTorch Lightning DataModule to dynamically load datasets.

    Loads dataset classes from the `datasets/` module based on name,
    and instantiates them with given arguments. Supports train, val, and test splits.

    Attributes:
        train_batch_size (int): Batch size for training.
        train_num_workers (int): Number of workers for training DataLoader.
        test_batch_size (int): Batch size for testing.
        test_num_workers (int): Number of workers for testing DataLoader.
        dataset_name (str): Name of the dataset module and class to load.
        kwargs (dict): Additional dataset-specific arguments.
    """

    def __init__(
        self,
        train_batch_size=64,
        train_num_workers=8,
        test_batch_size=64,
        test_num_workers=1,
        dataset_name=None,
        **kwargs
    ):
        """
        Initializes the DataInterface.

        Args:
            train_batch_size (int): Training batch size.
            train_num_workers (int): Number of workers for train loader.
            test_batch_size (int): Testing batch size.
            test_num_workers (int): Number of workers for test loader.
            dataset_name (str): Name of dataset class/module (e.g., 'my_dataset' → datasets/my_dataset.py → class MyDataset).
            **kwargs: Additional arguments passed to the dataset class.
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs

        self.load_data_module()  # Dynamically load dataset class

    def setup(self, stage=None):
        """
        Set up train, validation, and test datasets.

        Args:
            stage (str): One of 'fit', 'test', or None (setup all).
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')

        if stage == 'test' or stage is None:
            self.test_dataset = self.instancialize(state='test')

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.train_num_workers
        )

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.train_num_workers
        )

    def test_dataloader(self):
        """Returns the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.test_num_workers
        )

    def load_data_module(self):
        """
        Dynamically imports the dataset module and loads its class.

        Converts dataset_name like 'my_dataset' to:
        - file: datasets/my_dataset.py
        - class: MyDataset
        """
        image_name = ''.join([i.capitalize() for i in (self.dataset_name).split('_')])

        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), image_name)
        except Exception as e:
            raise ValueError('Invalid Dataset File Name or Invalid Class Name!') from e

    def instancialize(self, **other_args):
        """
        Instantiate a dataset using init arguments defined in its class.

        Args:
            **other_args: Arguments like 'state' to control train/val/test split.

        Returns:
            Dataset: Instantiated dataset object.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()

        args1 = {arg: self.kwargs[arg] for arg in class_args if arg in inkeys}
        args1.update(other_args)
        return self.data_module(**args1)
