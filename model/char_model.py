import torch
import torch.nn as nn
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.file_utils import WEIGHTS_NAME
from utils.utils import logging

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation

        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class Waveblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1], padding="same"):
        super().__init__()
        self.n = len(dilations)

        if padding == "same":
            padding = kernel_size // 2

        self.init_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.convs_tanh = nn.ModuleList([])
        self.convs_sigm = nn.ModuleList([])
        self.convs = nn.ModuleList([])

        for dilation in dilations:
            self.convs_tanh.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding * dilation, dilation=dilation),
                    nn.Tanh(),
                )
            )
            self.convs_sigm.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding * dilation, dilation=dilation),
                    nn.Sigmoid(),
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, 1))

    def forward(self, x):
        x = self.init_conv(x)
        res_x = x

        for i in range(self.n):
            x_tanh = self.convs_tanh[i](x)
            x_sigm = self.convs_sigm[i](x)
            x = x_tanh * x_sigm
            x = self.convs[i](x)
            res_x = res_x + x

        return res_x


class TweetCharModel(nn.Module):
    def __init__(self, config, from_pretrain=None, use_msd=True,
                 embed_dim=64, lstm_dim=64, char_embed_dim=32, ft_lstm_dim=32, n_models=1):
        super().__init__()
        self.config = config
        len_voc = config['len_voc']
        self.use_msd = use_msd

        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)

        self.proba_lstm = nn.LSTM(n_models * 2, ft_lstm_dim, batch_first=True, bidirectional=True)

        self.lstm = nn.LSTM(char_embed_dim + ft_lstm_dim * 2, lstm_dim, batch_first=True,
                            bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 4, lstm_dim),
            nn.ReLU(),
            nn.Linear(lstm_dim, 2),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, start_probas, end_probas):
        bs, T = input_ids.size()

        probas = torch.cat([start_probas.unsqueeze(2), end_probas.unsqueeze(2)], -1)
        probas_fts, _ = self.proba_lstm(probas)

        char_fts = self.char_embeddings(input_ids)


        features = torch.cat([char_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)

        features = torch.cat([features, features2], -1)

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits

    def save_pretrained(
            self,
            parallel_model,
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            **kwargs,
    ):
        if os.path.isfile(save_directory):
            logging.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logging.info(f"Model weights saved in {output_model_file}")

class WaveNet(nn.Module):
    def __init__(self, config, from_pretrain=None, use_msd=True, dilations=[1],
                 cnn_dim=64, char_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3,
                 use_bn=True):
        super().__init__()
        self.config = config
        len_voc = config['len_voc']
        self.use_msd = use_msd

        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)

        self.cnn = nn.Sequential(
            Waveblock(char_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size,
                      dilations=dilations),
            nn.BatchNorm1d(cnn_dim),
            Waveblock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 2),
            Waveblock(cnn_dim * 2, cnn_dim * 4, kernel_size=kernel_size, dilations=dilations),
            nn.BatchNorm1d(cnn_dim * 4),
        )

        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 4, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, start_probas, end_probas):
        bs, T = input_ids.size()

        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(input_ids)


        x = torch.cat([char_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1)  # [Bs x T x nb_ft]

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


class ConvNet(nn.Module):
    def __init__(self, config, from_pretrain=None, use_msd=True,
                 cnn_dim=64, char_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3,
                 use_bn=False):
        super().__init__()
        self.config = config
        len_voc = config['len_voc']
        self.use_msd = use_msd

        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)

        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)

        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 2, cnn_dim * 4, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size, use_bn=use_bn),
        )

        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 8, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, start_probas, end_probas):
        bs, T = input_ids.size()

        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(input_ids)


        x = torch.cat([char_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1)  # [Bs x T x nb_ft]

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits