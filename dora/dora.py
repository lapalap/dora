import os
import json
import glob
import torch
import warnings
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from torch_dreams.dreamer import Dreamer
from torch_dreams.auto_image_param import BaseImageParam, AutoImageParam
from torch_dreams.batched_objective import BatchedObjective
from torch_dreams.batched_image_param import BatchedAutoImageParam

import torchvision.transforms as transforms
from skimage import io, transform

from typing import Callable, Union
from .objectives import ChannelObjective
from .results import Result
from .forward_hook import ForwardHook
from .outlier_detection import OutlierDetector
from .reduction_methods import get_mean_along_last_2_dims
from .visualizer import OutlierVisualizer


class Dora:
    def __init__(
        self,
        model: nn.Module,
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

        # TODO: add transforms

        if storage_dir[-1] == "/":
            storage_dir = storage_dir[:-1]

        self.storage_dir = storage_dir
        self.make_folder_if_it_doesnt_exist(name=storage_dir)

    def make_folder_if_it_doesnt_exist(self, name):

        if name[-1] == "/":
            name = name[:-1]

        folder_exists = os.path.exists(name)

        if folder_exists == True:
            num_files = len(self.__get_filenames_in_a_folder(folder=name))
            if num_files > 0:
                UserWarning(f"Folder: {name} already exists and has {num_files} items")
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
        only_maximization,
        num_samples,
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
        #TODO add information about image_parameter and transformations
        data = {
            "experiment_name": experiment_name,
            "only_maximization": only_maximization,
            "num_samples": num_samples,
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
            fp.close()

        return overwrite_neurons

    def generate_signals(
        self,
        experiment_name,
        layer: nn.Module,
        objective_fn: Callable,
        progress: bool = True,
        neuron_idx: Union[list, int] = None,
        only_maximization: bool = True,
        batch_size = 16,
        num_samples=1,
        width=256,
        height=256,
        iters=150,
        image_parameter = None,
        image_transforms = transforms.Compose([transforms.Pad(2, fill=.5, padding_mode='constant'),
                                           transforms.RandomAffine((-15,15),
                                                                   translate=(0, 0.1),
                                                                   scale=(0.85, 1.2),
                                                                   shear=(-15,15),
                                                                   fill=0.5),
                                           transforms.RandomCrop((224, 224),
                                                                 padding=None,
                                                                 pad_if_needed=True,
                                                                 fill=0,
                                                                 padding_mode='constant')]),
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

        local_dreamer = Dreamer(model=self.model, quiet=True, device=self.device)
        if image_transforms is not None:
            local_dreamer.set_custom_transforms(image_transforms)
        if image_parameter is None:
            image_parameter = AutoImageParam(height= height,
                                               width = width,
                                               device = self.device,
                                               standard_deviation = 0.01)
        
        #TODO update batch procedure
        overwrite_neurons = self.check_and_write_config(
            experiment_name=experiment_name,
            only_maximization = str(only_maximization),
            num_samples=num_samples,
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

        # if 'only maximization' we generate only Activation-Maximisation signal
        if only_maximization:
            signatures = ['+']
        else:
            signatures = ['+', '-']
            
            
        task_list = [[idx, idx_sample, sign] for idx in neuron_idx for idx_sample in range(num_samples) for sign in signatures ]

        ## objective generator for each neuron
        def make_custom_func(layer_number=0, channel_number=0, maximisation = True):
            if maximisation:
                constant = 1.
            else:
                constant = -1.
            def custom_func(layer_outputs):
                loss = layer_outputs[layer_number][channel_number].norm()
                return -constant*loss

            return custom_func

        counter = 0
        while tqdm(counter < len(task_list), disable=not (progress), desc="Generating s-AMS"):

            internal_batch_size = min(batch_size, len(task_list) - counter)
            batched_objective = BatchedObjective(
                objectives=[make_custom_func(channel_number=idx,
                                             maximisation=sign == '+') for idx, idx_sample, sign in task_list[counter:counter + internal_batch_size]]
            )

            # if overwrite_neurons == False and os.path.exists(filename) == True:
            #     # print(
            #     #     f"skippping neuron index:{idx}, sample {idx_sample}, sign {sign}  because it already exists here: {filename} with the same generation config"
            #     # )
            #     # image = Image.open(filename)
            #
            #     continue
            # else:
            #     # if sign == '+':
            #     #     objective_fn.constant = 1
            #     # elif sign == '-':
            #     #     objective_fn.constant = -1

            ## set up a batch of trainable image parameters
            bap = BatchedAutoImageParam(
                batch_size=internal_batch_size,
                width=width,
                height=height,
                standard_deviation=0.01
            )

            image_param = local_dreamer.render(
                image_parameter=bap,
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
                custom_func=batched_objective,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )

            for i, [idx, idx_sample, sign] in enumerate(task_list[counter:counter + internal_batch_size]):
                result_batch[i].save(experiment_folder + "/" + f"{idx}_{idx_sample}{sign}.jpg")

            counter += internal_batch_size


class SignalDataset(torch.utils.data.Dataset):
    """Custom dataset class for loading the signals"""

    def __init__(self, root_dir, N_r, N_s, transform=None):
        """
        #TODO fill this
        """
        self.root_dir = root_dir
        self.transform = transform

        self.N_r = N_r
        self.N_s = N_s

        self.metainfo = {}

        for x in glob.glob(f"{root_dir}/*.jpg"):
            x = os.path.basename(x)
            # [neuron_id, sample_id, sign]
            self.metainfo[x] = [int(x[:-5].split('_')[0]),int(x[:-5].split('_')[1]), x[-5]]

        assert self.N_r*self.N_s*2 == len(self.metainfo.keys())


    def __len__(self):
        return len(self.metainfo.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = list(self.metainfo.keys())[idx]

        img_path = os.path.join(self.root_dir,
                                img_name)
        image = io.imread(img_path)

        if self.transform:
            sample = self.transform(image)

        return sample, self.metainfo[img_name]


def compute_distance(A: torch.Tensor):
    """
    A: tensor of shape [N_r, N_s, N_r,  2]
    
    """
    assert len(A.shape) == 4

    A = A.mean(axis = 1)

    Beta = A[:, :,0] - A[:, :, 1]
    Beta = Beta / torch.diagonal(Beta)

    return Beta * torch.sqrt(Beta.T/Beta)






