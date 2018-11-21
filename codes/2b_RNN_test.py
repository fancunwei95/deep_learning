import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from RNN_model import RNN_model,StatefulLSTM,LockedDropout


print (sys.argv)
no_hidden_units = sys.argv[1]
no_of_epochs = sys.argv[2]
sequence_length_train=sys.argv[3]
opt = sys.argv[4]
LR = sys.argv[5]
ID = sys.argv[6]

no_hidden_units = int(no_hidden_units)
no_of_epochs = int(no_of_epochs)
LR = float(LR)
sequence_length_train = int(sequence_length_train)

vocab_size = 100000
print ("ID: ", ID)
print ("Glove: ", 'False')
print ("vocab_size: ", vocab_size)
print ("sequence_length: ",sequence_length_train)
print ("no_hidden_units: ", no_hidden_units)
print ("optimizer: ", opt)
print ("LR: ", LR)
print ("no_of_epochs:" , no_of_epochs)


#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1


#model = RNN_model(vocab_size,no_hidden_units)
model = torch.load('rnn_'+str(ID)+'.model')
model.cuda()

batch_size = 200
#no_of_epochs = 6
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

test_accu = []


for epoch in range(10):
    sequence_length = int(50*(epoch+1))
    # ## test
    model.eval()
    
    epoch_acc = 0.0
    epoch_loss = 0.0
    
    epoch_counter = 0
    
    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)
    
    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_test[I_permutation[i:i+batch_size]]
        
        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()
    
        with torch.no_grad():
            loss, pred = model(data,target,train=False)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
    
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)
    
    test_accu.append(epoch_acc)
    
    time2 = time.time()
    time_elapsed = time2 - time1
    
    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time_elapsed))

