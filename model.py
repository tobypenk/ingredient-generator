import numpy as np
from utils import *
from model_functions import *
import random
import pprint

data = open('ings.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

parameters = model(data, ix_to_char, char_to_ix, verbose = True, num_iterations=1000000, n_a=50, vocab_size=43)