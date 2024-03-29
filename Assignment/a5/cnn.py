#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """ 1D-Convolution Network"""

    def __init__(self, embed_size: int = 50, m_word: int = 21,
                 k: int = 5, f: int = None):
        """ Init 1D-CNN

        @param embed_size: embedding size of char (dimensionality)
        @param k: kernel size, also called window size
        @param f: number of filters, should be embed_size of word
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size,
                                out_channels=f,
                                kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """ map from X_reshaped to X_conv_out

        @param X_reshaped: Tensor of char-level embedding with shape (max_sentence_length, batch_size, e_char, m_word),
                           where e_char = embed_size of char, m_word = max_word_length.
        @return X_conv_out: Tensor of word-level embedding with shape (max_sentence_length, batch_size, e_word)
        """

        X_conv = self.conv1d(X_reshaped)  # (max_sentence_length, batch_size, f, m_word-k+1), f:filter num = e_word
        X_conv_out = self.maxpool(F.relu(X_conv))  # (max_sentence_length, batch_size, e_word, 1)

        return torch.squeeze(X_conv_out, -1)

### END YOUR CODE
