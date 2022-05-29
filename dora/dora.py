import os
import torch
import warnings
import torch.nn as nn
from torch_dreams.dreamer import dreamer

from typing import Callable

warnings.simplefilter("default")


class Dora:
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        storage_dir=".dora/",
        delete_if_storage_dir_exists=False,
        device=None,
    ):
        """Handles all stuff dora related. Would require a storage_dir where it would store the synthetic activatiion
        maximization signals (s-AMS) as images which would be fed into self.model to collect activations.

        Note: It is highly recommended that you run DORA on a GPU and not on a CPU for fast performance.

        Args:
            model (nn.Module): the pytorch model you'd want to work on
            layer (nn.Module): model.some_layer
            storage_dir (str, optional): name of the folder where the s-AMS images would be kept. Defaults to '.dora/'.
            delete_if_storage_dir_exists (bool, optional): if True, will delete all the stuff that exists in storage_dir before using it. Defaults to False.
            device (str, optional): specifies the device to be used for the model, for example, 'cuda:0'. If set to None, it automatically looks for a device. Defaults to None.
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.layer = layer
        self.dreamer = dreamer(model=self.model, quiet=True, device=device)
        self.storage_dir = storage_dir

        self.make_folder(
            name=storage_dir, delete_if_storage_dir_exists=delete_if_storage_dir_exists
        )

    def make_folder(self, name, delete_if_storage_dir_exists=False):

        if name[-1] == "/":
            name = name[:-1]

        folder_exists = os.path.exists(name)

        if folder_exists == True:
            num_files = len(self.get_filenames_in_a_folder(folder=name))
            print(num_files)
            if num_files > 0:
                warnings.warn(
                    f"Folder: {name} already exists and has {num_files} items ,if you want to delete it then set delete_if_storage_dir_exists = True"
                )
        else:
            os.mkdir(name)

    def get_filenames_in_a_folder(self, folder: str):
        """
        returns the list of paths to all the files in a given folder
        """

        if folder[-1] == "/":
            folder = folder[:-1]

        files = os.listdir(folder)
        files = [f"{folder}/" + x for x in files]
        return files

    def run(self, progress: bool = False, objective_fn: Callable = None):
        """Would generate s-AMS for each neuron inside self.layer based on the objective_fn.

        Args:
            progress (bool, optional): Set to True if you want to see tqdm progress. Defaults to False.
            objective_fn (Callable, optional): The objective function based on which the s-AMS would be generated. See https://github.com/Mayukhdeb/torch-dreams#visualizing-individual-channels-with-custom_func for more info. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def show_results(self):
        """Generates a plotly plot from the results. Useful to see the outliers in a 2D space.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
