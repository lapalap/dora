from importlib_metadata import method_cache
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

d = Dora(model=model, image_transforms=my_transforms)

d.generate_signals(
    neuron_idx=[i for i in range(10)],
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
    experiment_name="model.avgpool", neuron_idx=[i for i in range(10)], method="PCA"
)

print(result.embeddings.shape)  ## shape:[len(neuron_idx), 2]
print(result.outlier_neuron_idx)  ## list of neuron indices which were outliers

## runs an interactive dash app on http://127.0.0.1:8050/
result.visualize()
