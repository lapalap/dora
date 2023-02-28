"""Tests for hello function."""
import pytest

from dora import Dora
from dora.objectives import ChannelObjective

import torchvision.models as models
import torchvision.transforms as transforms

model = models.resnet18(pretrained=True).eval()
my_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@pytest.mark.cpu
def test_functionality():
    d = Dora(model=model, image_transforms=my_transforms)

    d.generate_signals(
        neuron_idx=[i for i in range(20)],
        layer=model.avgpool,
        objective_fn=ChannelObjective(),
        width=224,
        height=224,
        iters=3,
        progress=True,
        experiment_name="model.avgpool",
        overwrite_experiment=True,  ## pick up from where you left off
    )

    d.generate_signals(
        neuron_idx=[i for i in range(30)],
        layer=model.avgpool,
        objective_fn=ChannelObjective(),
        width=224,
        height=224,
        iters=3,
        progress=True,
        experiment_name="model.avgpool",
        overwrite_experiment=True,  ## pick up from where you left off
    )