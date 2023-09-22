import pandas as pd
import numpy as np
import re

a = np.load('./data/text_vectors_v1.npy')
print(a.shape)
# tensor = pd.read_csv('./data/tensor-nltk.csv')

# print(tensor['Tensor'].values[0])
# a= tensor['Tensor'].values[0]
# tensor_google = np.zeros((8896, 768))
# for i in range(tensor['Tensor'].shape[0]):
#     t = re.split('[,\n\\[\\] ]+', tensor['Tensor'].values[i])
#     t = np.array(t[1:769])
#     tensor_google[i] = t
  
# print(tensor_google)  
# np.save('./data/tensor-nltk.npy',tensor_google)