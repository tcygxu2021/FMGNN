import pickle
import sys
import timeit

import numpy as np

import torch

from sklearn.decomposition import PCA

from model import FMGNN, Trainer, Tester


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def load_aaindex(files,splitnum=-1):
    nlist=[]
    alist=[]
    fopen = open(files)
    for line in fopen.readlines():
        line = str(line).replace("\n","")        
        nlist.append(line.split(' ',splitnum)[0])
        alist.extend(list(map(float,line.split(' ',splitnum)[1:])))
    fopen.close()
    return nlist,np.array(alist)


def conn_aminoacid(str,num,aaList):
    conval = [0]*num
    
    lenstr=len(str)
    for char in iter(str):
        if(char not in aaList):
            lenstr -= 1
        else:
            conval = [i + j for i, j in zip(conval, map(lambda x:int(x*10000),aaindex_dict[char]))]
    if(lenstr > 0):
        conval = list(map(lambda x:int(x/lenstr)/10000, conval))
    return conval


def getPreEmbeddWeight(aaFile,aaindx_fn,dim):
    """add the aaindex as protein feature"""
    aaindex_list = []    
    global aaindex_dict   

    aaList, featList = load_aaindex(aaFile,aaindex_fn)
    X1 = featList.reshape(20,aaindex_fn)
    pca = PCA(n_components=dim)
    pca.fit(X1) 
    pca_weight=pca.fit_transform(X1)   
   
    aaindex_dict = dict(zip(aaList,pca_weight.tolist()))
    for key in word_dict:
        aaindex_list.append(conn_aminoacid(key,dim,aaList))
    """ended add the aaindex modified by tangcy 20201021"""   
    return np.array(aaindex_list)



if __name__ == "__main__":
    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]    
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    ########for debug################################
#    DATASET='human'
#    radius='2'
#    ngram='3'
#    dim=10
#    layer_gnn=3
#    side=5
#    window=2*side+1
#    layer_cnn=3
#    layer_output=3
#    lr=float(1e-3)
#    lr_decay=float(0.5)
#    decay_interval=10
#    weight_decay=float(1e-6)
#    iteration=100    

#    setting = DATASET+'--radius'+radius+'--ngram'+ngram+'--dim'+str(dim)+'--layer_gnn'+str(layer_gnn)+'--window'+str(window)+'--layer_cnn'+str(layer_cnn)+'--layer_output'+str(layer_output)+'--lr'+str(lr)+'--lr_decay'+str(lr_decay)+'--decay_interval'+str(decay_interval)+'--weight_decay'+str(weight_decay)+'--iteration'+str(iteration)      
#################################################################################
    proc_name = 'FMGNN'
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    aaindex_fn = 544
    aaindex_dict = {}  #dict
    aaindex_feat = getPreEmbeddWeight('z_AAindex.txt',aaindex_fn,dim)
    pre_embedWeight = aaindex_feat.reshape(n_word,dim)
    
    
    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
   
    """Set a model."""
    torch.manual_seed(1234)
    model = FMGNN(n_fingerprint,n_word,dim,layer_gnn,window,layer_cnn,layer_output,pre_embedWeight).to(device)
    trainer = Trainer(model,n_fingerprint,n_word,dim,layer_gnn,window,layer_cnn,layer_output,lr,weight_decay)
    tester = Tester(model)

    """Output files."""
    file_AUCs = '../output/result/AUCs--' + setting + proc_name + '.txt'
    file_model = '../output/model/' + setting + proc_name
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tF1_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        
        AUC_dev = tester.test(dataset_dev)[0]
        
        AUC_test, precision_test, recall_test, f1_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,
                AUC_test, precision_test, recall_test, f1_test]
        tester.save_AUCs(AUCs, file_AUCs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, AUCs)))
