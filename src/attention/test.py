import torch
import torch.nn.functional as F
import numpy as np
import math

n_head=8
seq_len=256
dims=128
nums=n_head*seq_len*dims

scale_factor=1/(math.sqrt(dims))

data = []

for i in range(nums):
    data.append(i%5)



data = np.array(data)
data = data.reshape(n_head,seq_len,dims)
data = torch.tensor(data,dtype=torch.float32)


query=key=value=data


# result = F.scaled_dot_product_attention(q,k,v)
# print(result)

attn_weight = query @ key.transpose(-2, -1) * scale_factor
attn_weight = torch.softmax(attn_weight, dim=-1)
result = attn_weight @ value

print(result)

