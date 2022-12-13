import torch
import torch.nn as nn
from .architecture_blocks import DenseODSTBlock


class DenseODSTBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_trees,
        num_layers,
        tree_output_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=False,
        Module=ODST,
        **kwargs
    ):
        layers = []
        for i in range(num_layers):
            oddt = Module(
                input_dim,
                num_trees,
                tree_output_dim=tree_output_dim,
                flatten_output=True,
                **kwargs
            )
            input_dim = min(
                input_dim + num_trees * tree_output_dim, max_features or float("inf")
            )
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = (
            num_layers,
            num_trees,
            tree_output_dim,
        )
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = (
                    min(self.max_features, layer_inp.shape[-1]) - initial_features
                )
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(
                *outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim
            )
        return outputs


class NODEModel(nn.Module):
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

        super().__init__()

    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def subset(self, x):
        return x[..., : self.hparams.output_dim].mean(dim=-2)

    def data_aware_initialization(self, datamodule):
        """Performs data-aware initialization for NODE"""
        logger.info("Data Aware Initialization....")
        # Need a big batch to initialize properly
        alt_loader = datamodule.train_dataloader(batch_size=2000)
        batch = next(iter(alt_loader))
        for k, v in batch.items():
            if isinstance(v, list) and (len(v) == 0):
                # Skipping empty list
                continue
            # batch[k] = v.to("cpu" if self.config.gpu == 0 else "cuda")
            batch[k] = v.to(self.device)

        # single forward pass to initialize the ODST
        with torch.no_grad():
            self(batch)

    def _build_network(self):
        self.backbone = NODEBackbone(self.hparams)
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification

        self.head = utils.Lambda(self.subset)

    def extract_embedding(self):
        if self.hparams.embed_categorical:
            if self.backbone.embedding_cat_dim != 0:
                return self.backbone.embedding_layers
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used as a Categorical Encoder"
            )