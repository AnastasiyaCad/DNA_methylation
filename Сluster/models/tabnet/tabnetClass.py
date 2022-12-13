import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet


class TabNetModelClassification(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            cat_idxs,
            cat_dims,
            cat_emb_dim,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
            **kwargs):
        self.produce = False

        super().__init__()
        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )

    def forward(self, x):
        x, _ = self.tabnet(x)
        # if self.produce:
        #     x =
        return x
