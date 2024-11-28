import torch
import typing, warnings
from torch_pruning.pruner import MetaPruner


class TaylorStepPruner(MetaPruner):
    def set_storage(self):
        self.groups_imps = {}
        self.counter = {}

    def store_importance(self, add=True):

        for i, group in enumerate(self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types)):
            if self._check_pruning_ratio(group):  # check pruning ratio

                group = self._downstream_node_as_root_if_attention(group)
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group)

                # print(imp)
                if i not in self.groups_imps:
                    self.groups_imps[i] = imp
                else:
                    self.groups_imps[i] += imp
                if i not in self.counter:
                    self.counter[i] = 1
                else:
                    self.counter[i] += 1

        # print(self.groups_imps[0].sum())

    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        for i, group in enumerate(self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types)):
            if self._check_pruning_ratio(group):  # check pruning ratio
                ##################################
                # Compute raw importance score
                ##################################
                group = self._downstream_node_as_root_if_attention(group)
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)
                imp = self.groups_imps[i] / self.counter[i]
                if imp is None: continue

                ##################################
                # Compute the number of dims/channels to prune
                ##################################
                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    current_channels = self.DG.get_out_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                else:
                    current_channels = self.DG.get_in_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_in_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                ##################################
                # collect pruning idxs
                ##################################
                pruning_idxs = []
                _is_attn, qkv_layers = self._is_attn_group(group)
                group_size = current_channels // ch_groups
                # dims/channels
                if n_pruned > 0:
                    if (self.prune_head_dims and _is_attn) or (not _is_attn):
                        n_pruned_per_group = n_pruned // ch_groups
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        if n_pruned_per_group > 0:
                            for chg in range(ch_groups):
                                sub_group_imp = imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size  # offset
                                pruning_idxs.append(sub_pruning_idxs)
                else:  # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs.append(imp_argsort[:n_pruned])

                # num heads
                if _is_attn and self.prune_num_heads:  # Prune entire attn heads
                    target_head_pruning_ratio = self.get_target_head_pruning_ratio(qkv_layers[0])
                    n_heads_removed = self.num_heads[qkv_layers[0]] - int(
                        self.init_num_heads[qkv_layers[0]] * (1 - target_head_pruning_ratio))
                    if n_heads_removed > 0:
                        head_imp = imp.view(ch_groups, -1).mean(1)
                        for head_id in torch.argsort(head_imp)[:n_heads_removed]:
                            pruning_idxs.append(
                                torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))

                if len(pruning_idxs) == 0: continue
                pruning_idxs = torch.unique(torch.cat(pruning_idxs, 0)).tolist()
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs)

                if self.DG.check_pruning_group(group):
                    # Update num heads after pruning
                    if _is_attn and self.prune_num_heads and n_heads_removed > 0:
                        for dep, _ in group:
                            if dep.target.module in self.num_heads:
                                self.num_heads[dep.target.module] -= n_heads_removed
                    yield group

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        global_head_importance = {}  # for attn head pruning
        for i, group in enumerate(self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types)):
            if self._check_pruning_ratio(group):
                group = self._downstream_node_as_root_if_attention(
                    group)  # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                # imp = self.estimate_importance(group)
                # raw importance score
                imp = self.groups_imps[i] / self.counter[i]
                group_size = len(imp) // ch_groups
                if imp is None: continue
                if ch_groups > 1:
                    # Corresponding elements of each group will be removed together.
                    # So we average importance across groups here. For example:
                    # imp = [1, 2, 3, 4, 5, 6] with ch_groups=2.
                    # We have two groups [1,2,3] and [4,5,6].
                    # The average importance should be [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0)
                else:
                    # no grouping
                    dim_imp = imp
                global_importance.append((group, ch_groups, group_size, dim_imp))

                # pre-compute head importance for attn heads
                _is_attn, qkv_layers = self._is_attn_group(group)
                if _is_attn and self.prune_num_heads and self.get_target_head_pruning_ratio(qkv_layers[0]) > 0:
                    # average importance of each group. For example:
                    # the importance score of the group
                    # imp = [1, 2, 3, 4, 5, 6] with num_heads=2
                    # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                    # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                    head_imp = imp.view(ch_groups, -1).mean(1)  # average importance by head.
                    global_head_importance[group] = (qkv_layers, head_imp)

        if len(global_importance) == 0 and len(global_head_importance) == 0:
            return

        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################

        # Find the threshold for global pruning
        if len(global_importance) > 0:
            concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
            target_pruning_ratio = self.per_step_pruning_ratio[self.current_step]
            n_pruned = len(concat_imp) - int(
                self.initial_total_channels *
                (1 - target_pruning_ratio)
            )
            if n_pruned > 0:
                topk_imp, _ = torch.topk(concat_imp, k=n_pruned, largest=False)
                thres = topk_imp[-1]

        # Find the threshold for head pruning
        if len(global_head_importance) > 0:
            concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
            target_head_pruning_ratio = self.per_step_head_pruning_ratio[self.current_step]
            n_heads_removed = len(concat_head_imp) - int(
                self.initial_total_heads *
                (1 - target_head_pruning_ratio)
            )
            if n_heads_removed > 0:
                topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_removed, largest=False)
                head_thres = topk_head_imp[-1]

        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.DG.get_in_channels

            # Prune feature dims/channels
            pruning_indices = []
            if len(global_importance) > 0 and n_pruned > 0:
                if ch_groups > 1:  # re-compute importance for each channel group if channel grouping is enabled
                    n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
                    if n_pruned_per_group > 0:
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        _is_attn, _ = self._is_attn_group(group)
                        if not _is_attn or self.prune_head_dims == True:
                            raw_imp = self.estimate_importance(group)  # re-compute importance
                            for chg in range(
                                    ch_groups):  # determine pruning indices for each channel group independently
                                sub_group_imp = raw_imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size
                                pruning_indices.append(sub_pruning_idxs)
                else:
                    _pruning_indices = (imp <= thres).nonzero().view(-1)
                    imp_argsort = torch.argsort(imp)
                    if len(_pruning_indices) > 0 and self.round_to:
                        n_pruned = len(_pruning_indices)
                        current_channels = get_channel_fn(module)
                        n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                        _pruning_indices = imp_argsort[:n_pruned]
                    pruning_indices.append(_pruning_indices)

            # Prune heads
            if len(global_head_importance) > 0 and n_heads_removed > 0:
                if group in global_head_importance:
                    qkv_layers, head_imp = global_head_importance[group]
                    head_pruning_indices = (head_imp <= head_thres).nonzero().view(-1)
                    if len(head_pruning_indices) > 0:
                        for head_id in head_pruning_indices:
                            pruning_indices.append(
                                torch.arange(head_id * group_size, (head_id + 1) * group_size, device=head_imp.device))
                    for qkv_layer in qkv_layers:
                        self.num_heads[qkv_layer] -= len(head_pruning_indices)  # update num heads after pruning

            if len(pruning_indices) == 0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self.DG.check_pruning_group(group):
                yield group



