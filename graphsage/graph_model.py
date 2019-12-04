import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self,
                 num_nodes,
                 embedding_dim,
                 features_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type, 
                 **kwargs):
        super(GraphSAGE, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(num_nodes, embedding_dim=embedding_dim)
        self.layers = nn.ModuleList()
        
        in_feats = embedding_dim + features_dim
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, g):
        """ 'node_id' and 'features' are used from g.ndata. """
        ndata = g.ndata
        node_embeds = self.embedding(ndata["node_id"])
        h = torch.cat((node_embeds, ndata["features"]), dim=1)
        for layer in self.layers:
            h = layer(g, h)
        return h
    
    def get_hidden(self, g):
        ndata = g.ndata
        node_embeds = self.embedding(ndata["node_id"])
        h = torch.cat((node_embeds, ndata["features"]), dim=1)
        # remove last layer
        for layer in self.layers[:-1]: 
            h = layer(g, h)
        return h