from torch import nn
import torch
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
        quarter_hidden_size = int(half_hidden_size/2)
        self.linear_1 = nn.Linear(hidden_size,half_hidden_size)
        self.activation_1 = nn.ReLU()
        self.linear_2 = nn.Linear(half_hidden_size, n_classes)
        # self.activation_2 = nn.LeakyReLU(0.1)
        # self.linear_3 = nn.Linear(quarter_hidden_size,n_classes)

    def forward(self, **kwargs):
        out = self.uniter_model(**kwargs)
        out = self.uniter_model.pooler(out)
        if kwargs["gender_race_probs"] is not None:
            gender_race_probs = kwargs["gender_race_probs"]
            out = torch.cat((out, gender_race_probs), 1) # concatenate the uniter output with gender and race probabilities
        out = self.linear_1(out)
        out = self.activation_1(out)
        out = self.linear_2(out)
        # out = self.activation_2(out)
        # out = self.linear_3(out)
        
        return out
