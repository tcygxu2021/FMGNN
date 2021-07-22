import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from Lookahead import Lookahead
from RAdam import RAdam




class FMGNN(nn.Module):
    def __init__(self,n_fingerprint,n_word,dim,layer_gnn,window,layer_cnn,layer_output,pre_embedWeight):
        super(FMGNN, self).__init__()
        self.layer_gnn=layer_gnn
        self.layer_cnn=layer_cnn
        self.n_fingerprint=n_fingerprint
        self.n_word=n_word
        self.dim=dim
        self.window=window
        self.layer_output=layer_output
   
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.embed_word.weight.data.copy_(torch.from_numpy(pre_embedWeight))
        
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        """DNN for n_fingerprint*n_word"""    
        self.W_out_1 = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_output)])

        self.W_out_2 = nn.ModuleList([nn.Linear(4*dim, 4*dim)
                                      for _ in range(layer_output)])
    
        self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction_1 = nn.Linear(dim, 2)
        self.W_interaction_2 = nn.Linear(4*dim, 2)
        self.W_maxpools = nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cn = n_fingerprint
        self.k = 10
        self.linear_c = nn.Linear(self.cn, dim, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.cn))

        self.wn = n_word
        self.linear_p = nn.Linear(self.wn, dim, bias=True)
        self.wv = nn.Parameter(torch.randn(self.k, self.wn))
       
    def FM_C(self, x):
        linear_part = self.linear_c(x)
        # matrix multiplication: (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        # matrix multiplication: (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        
        return torch.unsqueeze(torch.mean(output, 0), 0)

    def FM_P(self, x):
        linear_part = self.linear_p(x)
        # matrix multiplication: (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.wv.t())  # out_size = (batch, k)
        # matrix multiplication: (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.wv, 2).t()) # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return torch.unsqueeze(torch.mean(output, 0), 0)

    def gnn(self, xs, A, layer):
  
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)   
        return torch.unsqueeze(torch.mean(xs, 0), 0)
 
    
    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))

        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

 
    def forward(self, inputs):

        fingerprints, adjacency, words = inputs
        
        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)

        fn_input = torch.nn.functional.one_hot(fingerprints, self.n_fingerprint)
        fn_input = fn_input.float()
        fn_c = self.FM_C(fn_input)
        vector_c = torch.cat((compound_vector, fn_c), 1)
        
        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)

        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, self.layer_cnn)
        fn_input = torch.nn.functional.one_hot(words, self.n_word)
        fn_input = fn_input.float() 
        fn_p = self.FM_P(fn_input)
        vector_p = torch.cat((protein_vector, fn_p), 1)

        cat_vector = torch.cat((vector_c, vector_p), 1)
        for j in range(self.layer_output):
           cat_vector = torch.relu(self.W_out_2[j](cat_vector))
        interaction = self.W_interaction_2(cat_vector)
        
        return interaction
    
    def add_loss(self,pred,cor):
        
        losses = cor * torch.log(pred) + (1 - cor) * torch.log(1 - pred)
        loss = -torch.sum(losses)

        
        return loss

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
 
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model,n_fingerprint,n_word,dim,layer_gnn,window,layer_cnn,layer_output,lr,weight_decay):
        self.model = model
        #self.optimizer = optim.Adam(self.model.parameters(),
        #                            lr=lr, weight_decay=weight_decay)
        optim1 = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        self.optimizer = Lookahead(optim1, alpha=0.5, k=10)

    def train(self, dataset):
        np.random.shuffle(dataset)

        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):

        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        #print("".join(str(i) for i in T))
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        f1 = f1_score(T, Y)
        return AUC, precision, recall, f1

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

