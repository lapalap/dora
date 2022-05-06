import dora
import torchvision

model = torchvision.models.resnet18(pretrained = True)

explorer = dora.Dora(model)

# getting layer names -- list of strings with names of layers
name_list = explorer.get_layers_names()
layer_name = name_list[-1]

# generate AMS (FVs)

ams = explorer.generate_ams(layer_name,
                           FV_method,
                           params,
                           )

# getting the embeddings
embeddings = explorer.get_embeddings(layer_name,
                                     ams)

# show atlas ( dimentionaluity reduction)
explorer.plot_atlas(embeddings)

# finding outliers
explorer.find_outliers(embeddings)
