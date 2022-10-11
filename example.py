import torch
import torchvision.models as models
import torchvision.transforms as transforms

from dora import Dora
from dora.objectives import ChannelObjective
from torch_dreams.auto_image_param import BaseImageParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neuron_indices = [i for i in range(5, 7)]

model = models.resnet18(pretrained=True).eval().to(device)
d = Dora(model=model, device=device)

d.generate_signals(
    neuron_idx=neuron_indices,
    layer=model.avgpool,
    image_parameter = None,
    image_transforms = transforms.RandomRotation((-15, 15)),
    objective_fn=ChannelObjective(),
    lr=18e-3,
    width=224,
    height=224,
    iters=90,
    experiment_name="model.avgpool",
    overwrite_experiment=True,  ## will still use what already exists if generation params are same
)


#
# d.collect_encodings(layer=model.avgpool, experiment_name="model.avgpool")
#
# result = d.run_outlier_detection(
#     experiment_name="model.avgpool",
#     neuron_idx=neuron_indices,
#     method="PCA",
#     outliers_fraction=0.1,
# )
#
# print(result.embeddings.shape)  ## shape:[len(neuron_idx), 2]
# print(result.outlier_neuron_idx)  ## list of neuron indices which were outliers
#
# ## runs an interactive dash app on http://127.0.0.1:8050/
# result.visualize()
#
# ## get outliers
# outliers = result.get_outlier_neurons()
# print(outliers)
#
# ## get normal neurons
# normal_neurons = result.get_outlier_neurons()
# print(normal_neurons)
