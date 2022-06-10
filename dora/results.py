from dataclasses import dataclass
import imp
import torch
from PIL import Image

from dataclasses import dataclass


@dataclass
class Result:
    """Wrapper to hold results from each Dora.generate_signals()"""

    s_ams: torch.tensor
    image: Image
    encoding: torch.tensor
