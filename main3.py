import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models.AlexNet import AlexNet
from models.ResNet50 import ResNet50
from models.ResNet18 import ResNet18
from models.Resnet18_last_meanpool import ResNet18_last_meanpool
from models.Resnet18_text_sum import ResNet18_text_sum
import matplotlib.pyplot as plt
from utils import set_random_seed
import os
import csv
from PIL import Image
import pandas as pd
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split, KFold

import numpy as np
set_random_seed(123)
np.random.seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')
os.system("ulimit -n 100000")
# =============================== EDIT HERE ===============================
"""
    Build model Architecture and do experiment.
"""
# lenet / alexnet / resnet50 /resnet18 / resnet18_text /resnet18_text_sum
model_name = 'resnet18_text_sum'
# cifar10 / svhn

# Hyper-parameters to search
# num_epochs_list = [1, 2, 3, 4, 5]
# learning_rate_list = [0.1, 0.2, 0.3]
# reg_lambda_list = [0.1, 0.2, 0.3]
# batch_size_list = [100]
# num_search = 10
# batch_size = 100

# LeNet
# num_epochs_list = [10, 20,50]
# learning_rate_list = [0.05, 0.01, 0.005, 0.02]
# reg_lambda_list = [0.1, 0.01, 0.001]
# batch_size_list = [50, 100, 200,500]

#AlexNet
# num_epochs_list = [10, 20, 50, 70]
# learning_rate_list = [0.005, 0.01, 0.02, 0.05]
# reg_lambda_list = [0.1, 0.01, 0.001]
# batch_size_list = [10,50, 100, 200]

# num_epochs_list = [1, 2, 3, 4, 5]
# learning_rate_list = [0.1, 0.2, 0.3]
# reg_lambda_list = [0.1, 0.2, 0.3]
# batch_size_list = [100]
# num_search = 10
# batch_size = 100

# epoch 30 , lr 0.0005, lambda 0.0001, batch 32  => accuracy 0.35

def main():
        
    num_epochs_list = [20]
    learning_rate_list = [0.0005]
    reg_lambda_list = [ 0.001]
    batch_size_list = [32]

    num_search = 1
    test_every = 1
    print_every = 1

    # batch normalization for ResNet18
    use_batch_norm = True
    # =========================================================================

    # Dataset


    np_posters = np.load('./data/Poster_images(3--)_v1.npy', allow_pickle=True)
    
    np_genres = pd.read_csv('./sampled_index_multihot.csv')
    text_vectors = np.load('./data/tensor-nltk.npy')
    
    a = np.array(np_genres.sum(axis=0)[1:].values, dtype = float)
    column = np_genres.columns
    normed_weight = [(1-x/a.sum()) for x in a]
    sample_index = np_genres['index']
    new_columns = [t for t in np_genres.columns if t != 'index']
    np_genres = np_genres[new_columns]
    
    
    
    
    np_genres = np.array(np_genres.values)
    sample_posters = np_posters[sample_index.values]
    
    # sample_text_vectors = text_vectors
    sample_text_vectors = np.random.randn(8896,768)
    print(sample_text_vectors.shape)
    sample_posters= np.array(sample_posters.tolist())
    
    
    print(sample_posters.shape)
    x, test_x, y,test_y, vector, test_vector  = train_test_split(sample_posters, np_genres,sample_text_vectors, train_size=0.9, shuffle=True)
    
 
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    test_vector = torch.Tensor(test_vector)
    normed_weight = torch.FloatTensor(normed_weight)
    
    use_normed_weight = True
    kf = KFold(n_splits=3, shuffle=True, random_state=42, )
    cv_accuracy =[]
    k=0
    for train_idx, valid_idx in kf.split(x):
        
        train_x = torch.Tensor(x[train_idx])
        train_vector = torch.Tensor(vector[train_idx])
        train_y = torch.Tensor(y[train_idx])
        valid_x = torch.Tensor(x[valid_idx])
        valid_vector = torch.Tensor(vector[valid_idx])
        valid_y = torch.Tensor(y[valid_idx])
        

        trainset = torch.utils.data.TensorDataset(train_x,train_vector,train_y)
        validset = torch.utils.data.TensorDataset(valid_x,valid_vector, valid_y)
        testset = torch.utils.data.TensorDataset(test_x, test_vector, test_y)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        print('Search Starts...')
        best_acc = -1
        best_hyper_params = []
        for search_cnt in range(num_search):
            num_epochs = random.choice(num_epochs_list)
            learning_rate = random.choice(learning_rate_list)
            reg_lambda = random.choice(reg_lambda_list)
            batch_size = random.choice(batch_size_list)

            
                
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
            validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
            
            if model_name == 'resnet18_last_meanpool':
                model = ResNet18_last_meanpool(3,22,learning_rate, reg_lambda, device, normed_weight, use_normed_weight)
            elif model_name == 'resnet18_text_sum':
                model = ResNet18_text_sum(3,22, learning_rate, reg_lambda, device, normed_weight, use_normed_weight)
            model = model.to(device)

            model.train_(trainloader,validloader,  num_epochs, test_every, print_every)

            # TEST ACCURACY
            model.restore()
            real_y, pred_y = model.predict(testloader)
            correct =0
            for i, predict in enumerate(pred_y):
                if real_y[i][predict] == 1:
                    correct+=1
            total = len(pred_y)
            test_acc = correct / total
     
            if test_acc > best_acc:
                best_acc = test_acc
                best_hyper_params = [num_epochs, learning_rate, reg_lambda, batch_size]
            print(f'search count: {search_cnt}, cur_test_acc: {test_acc},  best_test_acc: {best_acc}')
            if model_name == 'resnet18_last_meanpool':
                os.remove('./best_model/ResNet18_last_meanpool.pt') 
            elif model_name == 'resnet18_text_sum':
                os.remove('./best_model/ResNet18_text_sum.pt')    
            cv_accuracy.append(best_acc)
            model.plot_accuracy(k)
            k+=1
        print(f'best_valid_acc: {best_acc}, best_hyper_params: {best_hyper_params} (num_epochs, learning_rate, reg_lambda, batch_size, last_mean)')
    print(f'정확도 평균 : {np.mean(cv_accuracy)} ')   
    
if __name__ == '__main__':
    main()