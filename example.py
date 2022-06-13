import torch
import torchvision.models as models
import torchvision.transforms as transforms

from dora import Dora
from dora.objectives import ChannelObjective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neuron_indices = [i for i in range(100, 200)]

model = models.resnet18(pretrained=True).eval().to(device)
my_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

d = Dora(model=model, image_transforms=my_transforms, device=device)

d.generate_signals(
    neuron_idx=neuron_indices,
    layer=model.avgpool,
    objective_fn=ChannelObjective(),
    width=224,
    height=224,
    iters=200,
    experiment_name="model.avgpool",
    overwrite_experiment=True,  ## will still use what already exists if generation params are same
)

d.collect_encodings(layer=model.avgpool, experiment_name="model.avgpool")

result = d.run_outlier_detection(
    experiment_name="model.avgpool",
    neuron_idx=neuron_indices,
    method="PCA",
    outliers_fraction=0.1,
)

print(result.embeddings.shape)  ## shape:[len(neuron_idx), 2]
print(result.outlier_neuron_idx)  ## list of neuron indices which were outliers

## runs an interactive dash app on http://127.0.0.1:8050/
result.visualize()
