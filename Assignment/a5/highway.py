#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """ Highway network """

    def __init__(self, embed_size):
        """ Init Highway network

        @param embed_size: Embedding size of word (dimensionality), e_{word}
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of sentence

        @param X_conv_out (Tensor): Tensor with shape (max_sentence_length, batch_size, embed_size)
        @return X_highway (Tensor): combined output with shape (max_sentence_length, batch_size, embed_size)
        """

        X_proj = F.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = torch.mul(X_proj, X_gate) + torch.mul(X_conv_out, 1 - X_gate)

        return X_highway

### END YOUR CODE 

