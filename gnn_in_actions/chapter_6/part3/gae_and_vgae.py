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





def train(model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.)
    
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