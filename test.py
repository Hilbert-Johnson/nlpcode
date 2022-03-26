import torch
import torch.nn as nn
embedding = nn.Embedding(10, 3)
print(list(embedding.parameters()))
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input))
embedding = nn.Embedding(10, 3, padding_idx=0)
print(list(embedding.parameters()))
input = torch.LongTensor([[0,2,0,5]])
print(embedding(input))