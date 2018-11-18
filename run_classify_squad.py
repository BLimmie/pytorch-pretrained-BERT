from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_pretrained_bert
from pytorch_pretrained_bert.tokenization import printable_text, convert_to_unicode, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForLongClassification
from pytorch_pretrained_bert.optimization import BertAdam

def main():
    model = BertForLongClassification("bert-base-uncased")

if __name__ == "__main__":
    main()