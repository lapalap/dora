import os
import json
import torch
import warnings
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch_dreams.dreamer import dreamer
import torchvision.transforms as transforms

from typing import Callable, Union
from .objectives import ChannelObjective
from .results import Result
from .forward_hook import ForwardHook
from .outlier_detection import OutlierDetector
from .reduction_methods import get_mean_along_last_2_dims
from .visualizer import OutlierVisualizer

warnings.simplefilter("default")


class Dora:
    def __init__(
        self,
        model: nn.Module,
        image_transforms: Callable,
        storage_dir=".dora/",
        device="cpu",
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

        self.device = device
        self.model = model
        self.image_transforms = image_transforms
        self.dreamer = dreamer(model=self.model, quiet=True, device=device)

        if storage_dir[-1] == "/":
            storage_dir = storage_dir[:-1]

        self.storage_dir = storage_dir

        self.make_folder_if_it_doesnt_exist(name=storage_dir)

        self.results = {}

    def make_folder_if_it_doesnt_exist(self, name):

        if name[-1] == "/":
            name = name[:-1]

        folder_exists = os.path.exists(name)

        if folder_exists == True:
            num_files = len(self.__get_filenames_in_a_folder(folder=name))
            if num_files > 0:
                warnings.warn(
                    f"Folder: {name} already exists and has {num_files} items"
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

    def check_if_a_different_config_exists_with_same_name(self, filename, data):
        overwrite_neurons = False

        config_already_exists = os.path.exists(filename)
        if config_already_exists == True:
            existing_config = json.load(open(filename))
            for (k1, v1), (k2, v2) in zip(existing_config.items(), data.items()):

                assert (
                    k1 == k2
                ), f"Expected keys in config to be the same, but got {k1} and {k1}"

                if k1 != "neuron_idx":
                    ## if config fully matches, then do not overwrite existing neurons
                    if v1 == v2:
                        pass
                    else:
                        overwrite_neurons = True
                        break
        else:
            overwrite_neurons = True

        return overwrite_neurons

    def check_and_write_config(
        self,
        experiment_name,
        neuron_idx,
        width,
        height,
        iters,
        lr,
        rotate_degrees,
        scale_max,
        scale_min,
        translate_x,
        translate_y,
        weight_decay,
        grad_clip,
    ):
        data = {
            "experiment_name": experiment_name,
            "neuron_idx": neuron_idx,
            "width": width,
            "height": height,
            "iters": iters,
            "lr": lr,
            "rotate_degrees": rotate_degrees,
            "scale_max": scale_max,
            "scale_min": scale_min,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
        }
        folder_name = self.storage_dir + "/configs"

        self.make_folder_if_it_doesnt_exist(name=folder_name)
        filename = folder_name + "/" + experiment_name + ".json"

        overwrite_neurons = self.check_if_a_different_config_exists_with_same_name(
            filename=filename, data=data
        )

        ## if this is true then either the config does NOT already exists or exists with different params
        if overwrite_neurons == True:
            with open(filename, "w") as fp:
                json.dump(data, fp)

        return overwrite_neurons

    def generate_signals(
        self,
        experiment_name,
        layer: nn.Module,
        objective_fn: Callable,
        progress: bool = True,
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
        overwrite_experiment=False,
    ):
        """Would generate s-AMS for each neuron inside self.layer based on the objective_fn.

        Args:
            progress (bool, optional): Set to True if you want to see tqdm progress. Defaults to False.
            objective_fn (Callable, optional): The objective function based on which the s-AMS would be generated. See https://github.com/Mayukhdeb/torch-dreams#visualizing-individual-channels-with-custom_func for more info. Defaults to None.
        """

        ## config exists and matches - skip existing neurons
        ## config exists but does not match - overwrite existing neurons
        ## no config found - nothing
        overwrite_neurons = self.check_and_write_config(
            experiment_name=experiment_name,
            neuron_idx=neuron_idx,
            width=width,
            height=height,
            iters=iters,
            lr=lr,
            rotate_degrees=rotate_degrees,
            scale_max=scale_max,
            scale_min=scale_min,
            translate_x=translate_x,
            translate_y=translate_y,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )

        # time when execution started
        starting_time = time.time()
        sAMS_folder = self.storage_dir + "/sAMS"

        # creating a subfolder for sAMS (if not exists)
        folder_exists = os.path.exists(sAMS_folder)
        if folder_exists == False:
            os.mkdir(sAMS_folder)
            print(f"Subfolder for sAMS created at {sAMS_folder}")
        else:
            print(f"Using existing sAMS folder at {sAMS_folder}")
        # start generation
        experiment_name = (
            str(starting_time) if experiment_name is None else experiment_name
        )
        print(f"Experiment name: {experiment_name}")
        experiment_folder = sAMS_folder + "/" + experiment_name
        self.results[experiment_name] = {}

        # TODO: check if experiment folder exists (well, it shouldn't, but still)
        experiment_folder_exists = os.path.exists(experiment_folder)

        if not experiment_folder_exists:
            os.mkdir(experiment_folder)
        elif experiment_folder_exists == True and overwrite_experiment == True:
            print(f"Overwriting experiment: {experiment_name}")
        else:
            raise Exception(
                f"an experiment with the name {experiment_name} already exists, set overwrite_experiment = True if you want to overwrite it"
            )

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

            if overwrite_neurons == False and os.path.exists(filename) == True:
                print(
                    f"skippping neuron index:{idx} because it already exists here: {filename} with the same generation config"
                )
                image = Image.open(filename)

                # TODO: maybe not load all s-AMS to the memory -- only save them, and load?
                self.results[experiment_name][idx] = Result(
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

                self.results[experiment_name][idx] = Result(
                    s_ams=image_param.to_chw_tensor().unsqueeze(0),
                    image=transforms.ToPILImage()(image_param.to_chw_tensor()),
                    encoding=None,
                )

                image_param.save(filename=filename)

        # TODO: add logs to the experiment folder -- like hyperparameters information and etc..

        # self.collect_encodings(neuron_idx=neuron_idx)
        # self.run_outlier_detection(neuron_idx=neuron_idx)

    def load_results_from_folder(self, folder):
        raise NotImplementedError

    @torch.no_grad()
    def collect_encodings(self, layer, experiment_name, neuron_idx=None):

        # if neuron_idx is None, iterate over all results
        if neuron_idx is None:
            neuron_idx = list(self.results[experiment_name].keys())

        hook = ForwardHook(module=layer)

        for idx in tqdm(neuron_idx, desc="Collecting encodings"):
            input_tensor = self.results[experiment_name][idx].s_ams
            y = self.model.forward(input_tensor.to(self.device))

            self.results[experiment_name][idx].encoding = hook.output

        hook.close()

    def run_outlier_detection(
        self,
        experiment_name,
        neuron_idx=None,
        activation_reduction_fn: Callable = get_mean_along_last_2_dims,
        method="PCA",
        outliers_fraction=0.05,
        random_state=1,
    ):

        outlier_detector = OutlierDetector(
            name=method,
            outliers_fraction=outliers_fraction,
            random_state=random_state,
        )

        # if neuron_idx is None, iterate over all results
        if neuron_idx is None:
            neuron_idx = list(self.results[experiment_name].keys())

        encodings = torch.cat(
            [self.results[experiment_name][i].encoding for i in neuron_idx], dim=0
        )
        assert (
            encodings.ndim == 4
        ), f"Expected activations to have 4 dimensions [N, C, *, *] but got {encodings.ndim}"

        reduced_encodings = activation_reduction_fn(encodings)

        ## returns indices
        result = outlier_detector.run(activations=reduced_encodings)
        result_neuron_indices = np.array(neuron_idx)[result]

        return OutlierVisualizer(
            embeddings=outlier_detector.embeddings,
            outlier_neuron_idx=result_neuron_indices,
            neuron_idx=neuron_idx,
            experiment_name=experiment_name,
            storage_dir=self.storage_dir,
        )

    def show_results(self, experiment_name):
        """Generates a plotly plot from the results. Useful to see the outliers in a 2D space.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
