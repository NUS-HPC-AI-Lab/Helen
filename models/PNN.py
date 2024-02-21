# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from torch import nn
import torch
from models.base_model import BaseModel
from models.layers import MLP_Layer, EmbeddingLayer, InnerProductLayer


class PNN(BaseModel):
    def __init__(self,
                 feature_map,
                 task="binary_classification",
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 product_type="inner",
                 **kwargs):
        super(PNN, self).__init__(feature_map,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if product_type != "inner":
            raise NotImplementedError(f"product_type={product_type} has not been implemented.")
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="inner_product")
        input_dim = int(
            feature_map.num_fields * (feature_map.num_fields - 1) / 2) + feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.get_output_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True)
        self.reset_parameters()

    def forward(self, X):
        feature_emb = self.embedding_layer(X)
        inner_product_vec = self.inner_product_layer(feature_emb)
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_product_vec], dim=1)
        y_pred = self.dnn(dense_input)
        return y_pred