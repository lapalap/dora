import os
import torch
import warnings
import time
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch_dreams.dreamer import dreamer
import torchvision.transforms as transforms

from typing import Callable, Union
from .objectives import ChannelObjective
from .results import Result
from .forward_hook import ForwardHook
from .outlier_detection import UmapVisualizer
from .reduction_methods import get_mean_along_last_2_dims

warnings.simplefilter("default")


class Dora:
    def __init__(
        self,
        model: nn.Module,
        image_transforms: Callable,
        storage_dir=".dora/",
        delete_if_storage_dir_exists=False,
        device=None,
    ):
        """Handles all stuff dora related. Would require a storage_dir where it would store the synthetic Activatiion
        Maximization Signals (s-AMS) as images which would be fed into self.model to collect activations.

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
        self.image_transforms = image_transforms
        self.dreamer = dreamer(model=self.model, quiet=True, device=device)

        if storage_dir[-1] == "/":
            storage_dir = storage_dir[:-1]

        self.storage_dir = storage_dir

        self.__make_folder(
            name=storage_dir,
            delete_if_storage_dir_exists=delete_if_storage_dir_exists
        )

        self.results = {}

        self.umap = UmapVisualizer()

    def __make_folder(self, name, delete_if_storage_dir_exists=False):

        if name[-1] == "/":
            name = name[:-1]

        folder_exists = os.path.exists(name)

        if folder_exists == True:
            num_files = len(self.__get_filenames_in_a_folder(folder=name))
            print(num_files)
            if num_files > 0:
                warnings.warn(
                    f"Folder: {name} already exists and has {num_files} items ,if you want to delete it then set delete_if_storage_dir_exists = True"
                )
        else:
            os.mkdir(name)


    def __get_filenames_in_a_folder(self, folder: str):
        """
        returns the list of paths to all the files in a given folder
        """

        files = os.listdir(folder)
        files = [f"{folder}/" + x for x in files]
        return files

    def generate_signals(
        self,
        layer: nn.Module,
        progress: bool = False,
        objective_fn: Callable = None,
        neuron_idx: Union[list, int] = None,
        width=256,
        height=256,
        iters=150,
        lr=9e-3,
        rotate_degrees=15,
        scale_max=1.2,
        scale_min=0.8,
        translate_x=0.2,
        translate_y=0.2,
        weight_decay=1e-2,
        grad_clip=1.0,
        save_results=True,
        skip_if_exists=True,
        include_logs = True,
        experiment_name: None,
    ):
        """Would generate s-AMS for each neuron inside self.layer based on the objective_fn.

        Args:
            progress (bool, optional): Set to True if you want to see tqdm progress. Defaults to False.
            objective_fn (Callable, optional): The objective function based on which the s-AMS would be generated. See https://github.com/Mayukhdeb/torch-dreams#visualizing-individual-channels-with-custom_func for more info. Defaults to None.
        """

        #time when execution started
        starting_time = time.time()
        sAMS_folder = self.storage_dir + '/sAMS'

        #creating a subfolder for sAMS (if not exists)
        folder_exists = os.path.exists(sAMS_folder)
        if folder_exists == False:
            os.mkdir(sAMS_folder)
            print(
                f"Subfolder for sAMS created at {sAMS_folder}"
            )
        else:
            print(
                f"Using existing sAMS folder at {sAMS_folder}"
            )
        #start generation
        experiment_name = str(starting_time) if experiment_name is None
        print(
            f"Experiment name: {experiment_name}"
        )
        experiment_folder = sAMS_folder + '/' + experiment_name


        #TODO: check if experiment folder exists (well, it shouldn't, but still)
        experiment_folder_exists = os.path.exists(experiment_folder)
        os.mkdir(experiment_folder)

        if isinstance(neuron_idx, int):
            neuron_idx = [neuron_idx]
        else:
            assert (
                len(neuron_idx) > 0
            ), "Expected neuron_idx list to have a non zero length"

        for idx in tqdm(neuron_idx, disable=not (progress), desc="Generating s-AMS"):

            filename = experiment_folder + "/" + f"{idx}.jpg"

            if isinstance(objective_fn, ChannelObjective):
                objective_fn.channel_number = idx

            if (
                save_results == True
                and skip_if_exists == True
                and os.path.exists(filename) == True
            ):
                print(
                    f"skippping neuron index:{idx} because it already exists here: {filename}"
                )
                image = Image.open(filename)

                self.results[idx] = Result(
                    s_ams=self.image_transforms(image).unsqueeze(0),
                    image=image,
                    encoding=None,
                )
            else:
                image_param = self.dreamer.render(
                    layers=[layer],
                    width=width,
                    height=height,
                    iters=iters,
                    lr=lr,
                    rotate_degrees=rotate_degrees,
                    scale_max=scale_max,
                    scale_min=scale_min,
                    translate_x=translate_x,
                    translate_y=translate_y,
                    custom_func=objective_fn,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                )

                self.results[idx] = Result(
                    s_ams=image_param.to_chw_tensor().unsqueeze(0),
                    image=transforms.ToPILImage()(image_param.to_chw_tensor()),
                    encoding=None,
                )

                if save_results is True:
                    image_param.save(filename=filename)

        self.collect_encodings(neuron_idx=neuron_idx)
        self.run_outlier_detection(neuron_idx=neuron_idx)

    def load_results_from_folder(self, folder):
        raise NotImplementedError

    @torch.no_grad()
    def collect_encodings(self, neuron_idx=None):

        # if neuron_idx is None, iterate over all results
        if neuron_idx is None:
            neuron_idx = list(self.results.keys())

        hook = ForwardHook(module=self.layer)

        for idx in tqdm(neuron_idx, desc="Collecting encodings"):
            input_tensor = self.results[idx].s_ams
            y = self.model.forward(input_tensor)

            self.results[idx].encoding = hook.output

    def run_outlier_detection(
        self,
        neuron_idx=None,
        activation_reduction_fn: Callable = get_mean_along_last_2_dims,
    ):
        # if neuron_idx is None, iterate over all results
        if neuron_idx is None:
            neuron_idx = list(self.results.keys())

        encodings = torch.cat([self.results[i].encoding for i in neuron_idx], dim=0)
        assert (
            encodings.ndim == 4
        ), "Expected activations to have 4 dimensions [N, C, *, *] but got {encodings.ndim}"

        reduced_encodings = activation_reduction_fn(encodings)

        result = self.umap.run(activations=reduced_encodings)

    def show_results(self):
        """Generates a plotly plot from the results. Useful to see the outliers in a 2D space.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
