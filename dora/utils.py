from torch_dreams.auto_image_param import BaseImageParam
from torch_dreams.utils import init_image_param, normalize


class AdvBaseImageParam(BaseImageParam):
    def __init__(self, height, width, device):

        super().__init__()
        self.height = height
        self.width = width

        '''
        odd width is resized to even with one extra column
        '''
        if self.width % 2 == 1:
            self.param = init_image_param(height=self.height, width=width + 1, sd=0., device=device)
        else:
            self.param = init_image_param(height=self.height, width=self.width, sd=0., device=device)

        self.param.requires_grad_()
        self.optimizer = None

    def forward(self, device):
        return self.normalize(self.postprocess(device=device), device=device)

    def postprocess(self, device):
        img = torch.sigmoid(self.param)
        return img

    def normalize(self, x, device):
        return normalize(x=x, device=device)

    def fetch_optimizer(self, params_list, optimizer=None, lr=1e-3, weight_decay=0.):
        if optimizer is not None:
            optimizer = optimizer(params_list, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
        return optimizer