import os
import torch 

import numpy as np
import glob 

filename = "data/new_AMZN_electronics.npz"

data = np.load(filename)
loader = dict(data)
print(loader)

adj_matrix = torch.tensor(loader["adj_data"])
feature_matrix = torch.tensor(loader["attr_data"])
labels = loader["labels"]

class_names = loader.get("class_names")
metadata = loader.get("metadata")


print(f"Number of items {feature_matrix.shape[0]}")
print(f"Number of items = {adj_matrix.shape[0]} connected to {adj_matrix.shape[1]} items")
print(f"Number of items ={feature_matrix.shape[0]}, Number of Labels = {labels.shape[0]}")

feature_matrix[0, :]



# convert to sparse matrix
from torch_sparse import SparseTensor

edge_index = adj_matrix.nonzero(as_tuple=False).t()
edge_weight = adj_matrix[edge_index[0], edge_index[1]]
num_nodes = len(labels)

adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

print(edge_index[1])


# construct data object
from torch_geometric.data import Data
data = Data(x=feature_matrix, y=labels, adj_t=adj)
data 


# split for node prediction
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

transform = RandomLinkSplit()
transform(data)


def train(model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data)
    
    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label) + model.kl_loss()
    loss.backward()
    optimizer.step()
    
    return loss






@torch.no_grad()
def test(data):
    model.eval()
    z = model.encoder(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())



print(linear, variational)
best_val_auc = final_test_auc = 0
for epoch in range(1, 201):
    loss = train(model, criterion, optimizer)
    val_auc = test(val_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}")
print(f"Final Test: {final_test_auc:.4f}")