import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt

multi_hot = pd.read_csv('./data/moviegenre_multihotv1.csv', index_col=0)

multi_hot = multi_hot.drop(['News','Adult','Reality-TV','Game-Show','Talk-Show','Film-Noir'], axis='columns')

print(multi_hot.sum(axis=0))
idx_biography = multi_hot[multi_hot['Biography']==1].index
multi_hot = multi_hot.drop(idx_biography)
multi_hot = multi_hot.drop(['Biography'], axis='columns')

sampled_matrix = np.zeros((multi_hot.shape[0], multi_hot.shape[1]+1), dtype=object)


j=0
for i, row in enumerate(multi_hot.itertuples()):
    if row[1] == 1:
        sampled_matrix[j] = [t for t in row]
        j+=1
        if j== 1000: break
for i, row in enumerate(multi_hot.itertuples()):
    if row[1] ==1: continue
    if row[8] == 1:
        sampled_matrix[j] = [t for t in row]
        j+=1
        if j== 2000: break
for i, row in enumerate(multi_hot.itertuples()):
    if row[1] ==1 or row[8] ==1: continue
    if row[5] == 1:
        sampled_matrix[j] = [t for t in row]
        j+=1
        if j== 2500: break
for i, row in enumerate(multi_hot.itertuples()):
    if row[1] ==1 or row[8] ==1 or row[5] ==1: continue
    if row[11] == 1:
        sampled_matrix[j] = [t for t in row]
        j+=1
        if j== 3200: break
for i, row in enumerate(multi_hot.itertuples()):
    if row[1] ==1 or row[8] ==1 or row[5] ==1 or row[11] == 1: continue

    sampled_matrix[j] = [t for t in row]
    j+=1
    if j== 5000: break
print(sampled_matrix.sum(axis=0))
   
for i, row in enumerate(multi_hot.itertuples()):
    if row[7] ==1 or row[3] ==1 or row[1] ==1 or row[5] ==1 or row[8]==1 or row[11]==1: continue
    if i < sampled_matrix[j-1][0]: continue
    sampled_matrix[j] = [t for t in row]
    j+=1  



# for i, row in enumerate(multi_hot.itertuples()):
#     if i < sampled_matrix[j-1][0]: continue
#     if row[7] ==1 or row[3] ==1 or row[4] ==1 or row[11] ==1 or row[9]==1: continue
#     sampled_matrix[j] = [t for t in row]
#     j+=1     
#     if i>=26000: break
    
# for i, row in enumerate(multi_hot.itertuples()):
#     if i < sampled_matrix[j-1][0]: continue
#     if row[7] ==1 or row[3] ==1 or row[4] ==1 or row[11] ==1 or row[9]==1 or row[18] ==1: continue
#     sampled_matrix[j] = [t for t in row]
#     j+=1  


print(sampled_matrix[:j].shape[0])


new_columns = ['index']+[f for f in multi_hot.columns]
sampled_matrix = pd.DataFrame(sampled_matrix[:j], columns = new_columns)
sampled_matrix['Others']=0
for i in sampled_matrix:
    if sampled_matrix[i].sum() < 500:
        sampled_matrix['Others'] += sampled_matrix[i]
        sampled_matrix = sampled_matrix.drop(i, axis='columns')
        
idx_others = sampled_matrix[sampled_matrix['Others']>=1].index
sampled_matrix['Others'][idx_others] = 1
sampled_matrix.to_csv('sampled_index_multihot.csv', index=0)

print(sampled_matrix.sum(axis=0))



"""
1398
5530
6226
6509
13038
15751
16679
18195
18477
18629
18839
18957
19049
19214
19216
19223
20026
20241
20307
20323
20334
20341
20486
21051
21745
22290
22726
23068
23542
24006
24210
24468
24470
25354
25427
26343
26531
27602
27638
27784
28241
28342
28478
28573
28881
28911
28913
28914
28917
29319
29747
29823
30768
30965
30966
31002
31599"""