from functools import lru_cache
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import logging


class BaseModel(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_regularizer=0.,
                 net_regularizer=0.,
                 **kwargs):
        super(BaseModel, self).__init__()
        self._embedding_initializer = embedding_initializer
        self._feature_map = feature_map
        self.model_id = self.__class__.__name__ + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer

        self.feature_specs = self._feature_map.feature_specs
        # print class name
        logging.info(f"Use model {self.__class__.__name__}")

    def regularizer(self):
        embed_reg = net_reg = 0.
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "embedding_layer" in name:
                    embed_reg += self._embedding_regularizer / 2 * torch.norm(param, 2) ** 2
                else:
                    net_reg += self._net_regularizer / 2 * torch.norm(param, 2) ** 2
        reg_loss = embed_reg + net_reg
        self._embed_reg = embed_reg
        self._net_reg = net_reg
        self._reg = reg_loss
        return reg_loss

    @lru_cache(maxsize=1)
    def embed_params(self):
        embed_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "embedding_layer" in name and param.shape[-1] > 1:
                    embed_params.append(param)
        return embed_params

    @lru_cache(maxsize=1)
    def net_params(self):
        net_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "embedding_layer" not in name or param.shape[-1] == 1:
                    net_params.append(param)
        return net_params

    @lru_cache(maxsize=1)
    def get_feature_params_map(self):
        feature_params_map = dict()
        for feature, feature_spec in self.feature_specs.items():
            feature_params_map[feature] = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name and param.shape[-1] > 1:
                        if feature in name:
                            feature_params_map[feature].append(param)
        return feature_params_map

    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]:  # skip pretrained
                            continue
                        if self._embedding_initializer is not None:
                            try:
                                if v.padding_idx is not None:
                                    # the last index is padding_idx
                                    initializer = self._embedding_initializer.replace("(", "(v.weight[0:-1, :],")
                                else:
                                    initializer = self._embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError(
                                    f"embedding_initializer={self._embedding_initializer} is not supported."
                                )
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(reset_param)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def load_weights(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()

    @staticmethod
    def get_output_activation(task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return None
        else:
            raise NotImplementedError(f"task={task} is not supported.")

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        # logging.info(f"Total number of parameters: {total_params}.")
        return total_params

    def embed_params_norm(self):
        embed_params_sum = 0.
        for para in self.embed_params():
            embed_params_sum += torch.sum(para ** 2).item()
        embed_params_number = self.count_parameters(count_embedding=True) - self.count_parameters(count_embedding=False)
        return embed_params_sum / embed_params_number

    def net_params_norm(self):
        net_params_sum = 0.
        for para in self.net_params():
            net_params_sum += torch.sum(para ** 2).item()
        net_params_number = self.count_parameters(count_embedding=False)
        return net_params_sum / net_params_number

    def embed_grad_norm(self):
        embed_grad_sum = 0.
        for para in self.embed_params():
            embed_grad_sum += torch.sum(para.grad ** 2).item()
        embed_grad_number = self.count_parameters(count_embedding=True) - self.count_parameters(count_embedding=False)
        return embed_grad_sum / embed_grad_number

    def net_grad_norm(self):
        net_grad_sum = 0.
        for para in self.net_params():
            net_grad_sum += torch.sum(para.grad ** 2).item()
        net_grad_number = self.count_parameters(count_embedding=False)
        return net_grad_sum / net_grad_number
