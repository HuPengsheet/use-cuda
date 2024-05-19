import numpy as np

data=[]

for i in range(2048):
    data.append(i%10)


mean = sum(data) / len(data)  

  


variance = np.var(data)  

  

# 步骤 3: 归一化列表  

normalized_data = [(x - mean) / np.sqrt(variance) for x in data]  

  

print("均值:", mean)  

print("方差:", variance)  

print("归一化后的数据:", normalized_data)