import torch
import torch.nn as nn
from better_lstm import LSTM
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from daem.helpers import performance_batch

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)