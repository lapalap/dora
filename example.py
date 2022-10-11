import torch
import torchvision.models as models
import torchvision.transforms as transforms

from dora import Dora
from dora.objectives import ChannelObjective
from torch_dreams.auto_image_param import BaseImageParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neuron_indices = [i for i in range(5, 6)]

model = models.resnet18(pretrained=True).eval().to(device)
d = Dora(model=model, device=device)

# lucent.optvis.transform.pad(2, mode='constant', constant_value=.5),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.jitter(jttr),
#               lucent.optvis.transform.random_scale([0.995**n for n in range(-5,80)] + [0.998**n for n in 2*list(range(20,40))]),
#               lucent.optvis.transform.random_rotate(list(range(-20,20))+list(range(-10,10))+list(range(-5,5))+5*[0]),
#               lucent.optvis.transform.jitter(2),
#               torchvision.transforms.RandomCrop((224, 224), padding=None, pad_if_needed=True, fill=0, padding_mode='constant')

d.generate_signals(
    neuron_idx=neuron_indices,
    num_samples = 1,
    layer=model.fc,
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
    objective_fn=ChannelObjective(),
    lr=18e-3,
    width=224,
    height=224,
    iters=100,
    experiment_name="model.fc",
    overwrite_experiment=True,  ## will still use what already exists if generation params are same
)
from dora import SignalDataset

data = SignalDataset('/Users/kirillbykov/Documents/GitHub/dora/.dora/sAMS/model.fc',
                     N_r = 2,
                     N_s = 5,
                     transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])])
                     )


A = torch.zeros([2, 5, 2, 2])
with torch.no_grad():
    for i in data:
        print(i[0].shape)
        r = i[1][0] - 5
        s = i[1][1]
        sign = 0 if i[1][2] == "+" else 1
        out = model(i[0].view([1,3,224,224]))[0][5:7]
        A[r, s, sign, :] = out

print(A)

from dora import compute_distance

print(compute_distance(A))


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
