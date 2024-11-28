import torch
import typing, warnings
import torch.nn as nn
import numpy as np
from torch_pruning.pruner import function
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler
from torch_pruning.dependency import Group
from torch_pruning import ops
from torch_pruning.pruner import MetaPruner
from torch_pruning.pruner.importance import GroupNormImportance
from scipy.spatial import distance


class FPGMImportance(GroupNormImportance):
    def __init__(self,
                 p: int = 2,
                 group_reduction: str = "mean",
                 normalizer: str = 'mean',
                 bias=False,
                 target_types: list = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        super(FPGMImportance, self).__init__(p, group_reduction, normalizer, bias, target_types)

        self.step = 1

    @torch.no_grad()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                if self.step == 1:
                    local_imp = w.abs().pow(self.p).sum(1)
                else:
                    w = w.cpu().numpy()
                    similar_matrix = distance.cdist(w, w, 'euclidean')
                    local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                if self.step == 1:
                    local_imp = w.abs().pow(self.p).sum(1)
                else:
                    w = w.cpu().numpy()
                    similar_matrix = distance.cdist(w, w, 'euclidean')
                    local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)

                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    if self.step == 1:
                        local_imp = w.abs().pow(self.p)
                    else:
                        w = w.cpu().numpy().reshape(w.shape[0], 1)
                        # print(w.shape)
                        similar_matrix = distance.cdist(w, w, 'euclidean')
                        local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))
                    # local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        w = layer.bias.data[idxs]
                        if self.step == 1:
                            local_imp = w.abs().pow(self.p)
                        else:
                            w = w.cpu().numpy().reshape(w.shape[0], 1)
                            similar_matrix = distance.cdist(w, w, 'euclidean')
                            local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))

                        # local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    if self.step == 1:
                        local_imp = w.abs().pow(self.p)
                    else:
                        w = w.cpu().numpy().reshape(w.shape[0], 1)
                        similar_matrix = distance.cdist(w, w, 'euclidean')
                        local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        w = layer.bias.data[idxs]
                        if self.step == 1:
                            local_imp = w.abs().pow(self.p)
                        else:
                            w = w.cpu().numpy().reshape(w.shape[0], 1)
                            similar_matrix = distance.cdist(w, w, 'euclidean')
                            local_imp = torch.tensor(np.sum(np.abs(similar_matrix), axis=0))
                        # local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class FPGMPruner(MetaPruner):
    def __init__(self,
                 model: nn.Module,
                 example_inputs: torch.Tensor,  # a dummy input for graph tracing. Should be on the same
                 importance: typing.Callable,  # tp.importance.Importance for group importance estimation
                 global_pruning: bool = False,  # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html.
                 pruning_ratio: float = 0.3,  # total channel/dim pruning ratio, also known as pruning ratio
                 similar_pruning_ratio: float = 0.1,  # pruning ratio of similar criterion
                 customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
                 unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
                 ignored_layers: typing.List[nn.Module] = None):

        super(FPGMPruner, self).__init__(model=model,
                                         example_inputs=example_inputs,
                                         importance=importance,
                                         global_pruning=global_pruning,
                                         iterative_steps=1,
                                         pruning_ratio=pruning_ratio - similar_pruning_ratio,
                                         customized_pruners=customized_pruners,
                                         unwrapped_parameters=unwrapped_parameters,
                                         ignored_layers=ignored_layers)

        # self.pruning_ratio = pruning_ratio - similar_pruning_ratio
        self.similar_pruning_ratio = similar_pruning_ratio

        assert self.pruning_ratio >= 0, 'similar_pruning_ratio must smaller than pruning ratio'
        assert isinstance(self.importance, FPGMImportance), 'FPGMPruner only support FPGMImportance'

    def step(self, interactive=False):
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local
        print('step1 pruned: ', self.pruning_ratio)
        for group in pruning_method():
            group.prune()
        self.importance.step += 1
        self.pruning_ratio = self.similar_pruning_ratio / (1 - self.pruning_ratio)
        print('step2 pruned: ', self.pruning_ratio)
        for group in pruning_method():
            group.prune()




