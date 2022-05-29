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

d = Dora(model=model, layer=model.layer4, image_transforms=my_transforms)

d.run(
    neuron_idx=[i for i in range(512)],
    objective_fn=ChannelObjective(),
    width=224,
    height=224,
    iters=1,
    progress=True,
    save_results=True,
    skip_if_exists=True,
)

print(d.results)
