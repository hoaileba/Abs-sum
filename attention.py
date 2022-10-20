## Attention 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional

class Attention(nn.Module):
  def __init__(self,dim, query_dim):
    super(Attention,self).__init__()
    self.query_dim = query_dim
    self.embedding_dim = dim
    self.q = nn.Parameter(torch.empty(query_dim).uniform_(-0.1, 0.1))
    self.projection = nn.Linear(dim, query_dim)


  def forward(self,embedded_vector):
    """
      input:
        embedded_vector: 
          gồm nhiều vector của từ hoặc của các câu
          shape : (batch_size, N, embedding_dim)
      output:
        attention_vector :
          vector cuối cùng đại diện của feature của 1 câu, hoặc 1 đoạn
          shape: (batch_size, 1, embedding_dim)
    """
    # pass
    # print(self.q.shape, self.projection)
    attention_weight = torch.matmul(torch.tanh(self.projection(embedded_vector)), self.q )
    # print("attention_weight: ",attention_weight.shape)
    attention_weight = F.softmax(attention_weight, dim = 1).unsqueeze(dim=1)
    # print(attention_weight.shape, embedded_vector.shape)
    attention_vec = torch.bmm(attention_weight, embedded_vector).squeeze(dim=1)
    # print(attention_vec.shape)
    return attention_vec




# embedded_vec = torch.rand(5,20, 300)
# # print("embedded_vec: ", embedded_vec.shape)
# attention = Attention(300,2)
# x = attention(embedded_vec)