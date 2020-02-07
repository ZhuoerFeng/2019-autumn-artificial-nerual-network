from cotk import metric
from cotk.metric import NgramFwBwPerplexityMetric
from cotk.metric import BleuCorpusMetric, BleuPrecisionRecallMetric, SelfBleuCorpusMetric
from cotk.metric import accuracy
import torch 
import torch.nn as nn


class SuperviseGenModel(object):
    
    def __init__(self, inputs, vocab, gamma, lambda_g, hparams=None):
        pass

    def foward(self, inputs):
        # embedder = 
        # encoder = 
        embeds = nn.Embedding(2, 5)
        


        pass

    def close(self):
        pass