from torch import nn

from model.model import UniterModel


class MemeUniter(nn.Module):

    def __init__(self,
                 uniter_model: UniterModel,
                 hidden_size: int,
                 n_classes: int):
        super().__init__()
        self.uniter_model = uniter_model
        self.n_classes = n_classes
        #Added an extra linear layer for classification
        half_hidden_size = int(hidden_size/2)
        self.linear_1 = nn.Linear(hidden_size,half_hidden_size)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(half_hidden_size, n_classes)

    def forward(self, **kwargs):
        out = self.uniter_model(**kwargs)
        out = self.uniter_model.pooler(out)
        out = self.linear_1(out)
        out = self.activation(out)
        out = self.linear_2(out)
        return out
