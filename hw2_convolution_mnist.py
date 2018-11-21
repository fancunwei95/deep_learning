# Cunwei Fan

class ConvolutionNetwork:
    
    def __init__(self,filter_shape,shape_array,xtrain,ytrain):
        # shape_array is a 1D array each term means the node number of each layer
        # please include the input layer and output layer as well
        
        self.filter_shape=np.copy(filter_shape)
        self.layer_shape =np.copy(shape_array)
    
        self.dim0 = self.filter_shape[0]
        
        self.dim1 = (xtrain.shape[1]-self.filter_shape[1])+1
        self.dim2 = (xtrain.shape[2]-self.filter_shape[2])+1
        self.dim12 = self.dim1*self.dim2
        
        self.layer_shape = np.hstack([self.dim0*self.dim12,self.layer_shape])
        self.layer_num = self.layer_shape.shape[0]
        

        self.K_matrix = 0.01*np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2])
        self.W_matrix_dict ={i:0.01*(np.random.randn(self.layer_shape[i],self.layer_shape[i+1])) \
                            for i in range(self.layer_num-1)}
        self.b_matrix_dict ={i:0.01*np.random.randn(1,self.layer_shape[i+1]) \
                            for i in range(self.layer_num-1)}
        
        self.dK = np.zeros(self.filter_shape)
        self.dW_dict = {i:np.zeros((shape_array[i],self.layer_shape[i+1])) for i in range(1,self.layer_num-1)}
        self.db_dict = {i:np.zeros((1,self.layer_shape[i+1])) for i in range(self.layer_num-1)}
        
        self.xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2])
        self.ytrain = ytrain
        self.costs=[]
        self.costs_step=[]
        self.epoch_num_trained = 0
        image_shape = (xtrain.shape[1],xtrain.shape[2])
        back_filter_shape = (self.dim1,self.dim2)
        self.forward_convol_idx = self.get_index(image_shape, (filter_shape[1],filter_shape[2]) )
        self.backward_convol_idx = self.get_index(image_shape, back_filter_shape)
        
    def get_index(self,image_shape,filter_shape):
        dim1 = image_shape[0]-filter_shape[0]+1
        dim2 = image_shape[1]-filter_shape[1]+1
        iters = product(range(dim1),range(dim2))
        index = np.zeros((dim1*dim2,filter_shape[0]*filter_shape[1]))
        k = 0
        for i,j in iters:
            a = np.array([x for x in product(np.arange(i,i+filter_shape[0]),np.arange(j,j+filter_shape[1]))])
            index[k] = np.ravel_multi_index([a[:,0],a[:,1]],image_shape)
            k+=1
        index = index.astype(int)
        return index
        
        
    def activation(self,x):
        #return np.exp(x)/(1.0+np.exp(x))
        return np.where(x>0,x,0)

    def activation_der(self,y):
        #return y*(1-y)
        return np.where(y>0,1,0)
    
    def convolve(self,X,K):
        dim1 = X.shape[0]-K.shape[0]+1
        dim2 = X.shape[1]-K.shape[1]+1
        iters = product(range(dim1),range(dim2))
        results = np.zeros((dim1,dim2))
        for i,j in iters:
            results[i,j] = np.sum(X[i:i+K.shape[0],j:j+K.shape[1]]*K[:,:])
        return results
    
    
    def forward_convolve(self,X,K):
        index = self.forward_convol_idx
        x = X[:,index.flatten()].reshape((X.shape[0],index.shape[0],index.shape[1]))
        result = np.tensordot(x,K.T,axes=1)
        result = np.swapaxes(result,1,2)
        result = result.reshape(result.shape[0],result.shape[1]*result.shape[2])
        return result
    
    def backward_convolve(self,X,K):
        index = self.backward_convol_idx
        x = X[index.flatten()].reshape(index.shape)
        result = x.dot(K.T)
        result = (result.T).reshape(self.dim0,self.filter_shape[1],self.filter_shape[2])
        return result
    
    #def backward_convolve(self,X,K):
    #    index = self.backward_convol_idx
    #    x = X[:,index.flatten()].reshape(X.shape[0],index.shape[0],index.shape[1])
    #    x = np.swapaxes(x,0,1)
    #    k = K.reshape((self.dim12,self.dim0,K.shape[0]),order='F')
    #    k = np.swapaxes(k,1,2)
    #    k = np.swapaxes(k,0,1)
    #    result = np.tensordot(x,k,axes=2)
    #    result = (result.T).reshape(self.dim0,self.filter_shape[1],self.filter_shape[2])
    #    return result
    
    
    def predict(self,xbatch,activate):
        # xbatch each row is a sample
        N = self.layer_num
        self.layer_value = {i:np.zeros((xbatch.shape[0],self.layer_shape[i]))\
                            for i in range(0,self.layer_num)}
        K = self.K_matrix
        K = K.reshape(K.shape[0],K.shape[1]*K.shape[2])
        self.layer_value[0] = self.forward_convolve(xbatch,K)
        self.layer_value[0] = activate(self.layer_value[0])
        for i in range(N-2):
            W = self.W_matrix_dict[i]
            b = self.b_matrix_dict[i]
            self.layer_value[i+1] = activate(self.layer_value[i].dot(W)+b)
        W = self.W_matrix_dict[N-2]
        b = self.b_matrix_dict[N-2]
        final_layer = np.exp(self.layer_value[N-2].dot(W)+b)
        final_layer = final_layer/(np.sum(final_layer,axis=1,keepdims=True))
        self.layer_value[N-1] = final_layer
        return final_layer
    
    def cost_func(self,xbatch,ybatch,activate):
        predict = self.predict(xbatch,activate)
        predict = np.log(predict)
        cost = -1.0*np.sum(predict*ybatch)/xbatch.shape[0]
        return cost
    
    def back_propogation(self,xbatch,ybatch,activation_der):
        # last layer:
        N = self.layer_num
        dY= self.layer_value[N-1] - ybatch
        SampleSize = 1.*xbatch.shape[0]
        self.dW_dict[N-2] = (self.layer_value[N-2].T).dot(dY)/SampleSize
        self.db_dict[N-2] = np.sum(dY,axis=0,keepdims=True)/SampleSize
        for j in range(N-2):
            i = N-3-j
            dY = dY.dot(self.W_matrix_dict[i+1].T)*activation_der(self.layer_value[i+1])
            self.dW_dict[i] = (self.layer_value[i].T).dot(dY)/SampleSize
            self.db_dict[i] = np.sum(dY,axis=0,keepdims=True)/SampleSize
        
        part1 = dY.dot(self.W_matrix_dict[0].T)*activation_der(self.layer_value[0])
        part1 = np.reshape(part1,(part1.shape[0],self.dim0,self.dim1*self.dim2))
        dK = np.empty((xbatch.shape[0],self.K_matrix.shape[0],self.K_matrix.shape[1],self.K_matrix.shape[2]))
        for i in range(part1.shape[0]):
            dK[i] = self.backward_convolve(xbatch[i],part1[i])
        self.dK = np.sum(dK,axis=0)/SampleSize
        return 
    
    def mini_batch_train(self,batch_size,alpha = 0.1,stepNum=1000):
        train_data_size = self.xtrain.shape[0]
        batch_size = batch_size
        shuffle_index = np.arange(train_data_size)
        
        stratify_index = [int(i*batch_size) for i in range(int(np.floor(train_data_size/batch_size)) )]
        stratify_index.append(train_data_size)
        cycle = len(stratify_index)-1
        N = self.layer_num
        
        for epoch in range(stepNum):
            costs = np.zeros(cycle)
            np.random.shuffle(shuffle_index)
            shuffledX = self.xtrain[shuffle_index,:]
            shuffledY = self.ytrain[shuffle_index,:]
            Alpha = alpha*np.exp(-self.epoch_num_trained*1.0/120.0)
            for i in range(cycle):
                index = np.arange(stratify_index[i%cycle],stratify_index[i%cycle+1])
                xbatch = shuffledX[index,:]
                ybatch = shuffledY[index,:]
                loss = self.cost_func(xbatch,ybatch,self.activation)
                self.back_propogation(xbatch,ybatch,self.activation_der)
                self.W_matrix_dict = {i:self.W_matrix_dict[i]-Alpha*self.dW_dict[i] for i in range(N-1)}
                self.b_matrix_dict = {i:self.b_matrix_dict[i]-Alpha*self.db_dict[i] for i in range(N-1)}
                self.K_matrix += -Alpha*self.dK
                costs[i] = loss
            cost = np.sum(costs)/cycle
            self.epoch_num_trained +=1
            if (self.epoch_num_trained)%(5) == 0 or epoch==0:
                print(str(self.epoch_num_trained)+": loss, "+ str(cost))
                self.costs.append(cost)
                self.costs_step.append(self.epoch_num_trained)
        self.costs.append(cost)
        self.costs_step.append(self.epoch_num_trained)
        return 1
    
    def get_accuracy(self,xtest,ytest,activation,test=True):
        mypredict_test = self.predict(xtest,activation)
        mypredict_test = np.argmax(mypredict_test,axis=1)
        accu = np.sum(1.*(mypredict_test == ytest.flatten()) )/mypredict_test.shape[0]
        if test:
            self.test_accu.append(accu)
        else:
            self.train_accu.append(accu)
        return accu
    

if __name__ == '__main__':

    import h5py
    import numpy as np
    from itertools import product
    import time
    import matplotlib.pyplot as plt

    f = h5py.File('MNISTdata.hdf5', 'r')
    xtest = np.array(f["x_test"])
    xtrain =np.array(f["x_train"])
    ytest = np.array(f["y_test"])
    ytrain = np.array(f["y_train"])
    f.close()

    def one_hot(y):
        N = y.shape[0]
        yresult = np.zeros((N,10))
        yresult[np.arange(N),y.flatten()] = 1.0
        return yresult

    def go_back_from_one_hot(y):
        a = np.arange(10)
        result = y.dot(a)
        result = np.reshape(result,(result.shape[0],1))
        return result
    ytrain= one_hot(ytrain)
    xtrain= np.reshape(xtrain,(xtrain.shape[0],28,28))

    filter_array = np.array([3,3,3])
    network = np.array([10])
    mynet = ConvolutionNetwork(filter_array,network,xtrain,ytrain)

    start = time.time()
    mynet.mini_batch_train(100,stepNum=50)
    dura = time.time()-start
    print ("each epoch take "+str(dura/50.0)+" seconds on average" )



    mypredict_train = mynet.predict(mynet.xtrain,mynet.activation)
    mypredict_train = np.argmax(mypredict_train,axis=1)
    ytrain_ori = go_back_from_one_hot(ytrain)
    accu = np.sum(1.*(mypredict_train == ytrain_ori.flatten()) )/mypredict_train.shape[0]
    print ("train accuracy: "+str(accu*100.0)+"%")


    mypredict_test = mynet.predict(xtest,mynet.activation)
    mypredict_test = np.argmax(mypredict_test,axis=1)
    accu = np.sum(1.*(mypredict_test == ytest.flatten()) )/mypredict_test.shape[0]
    print ("test accuracy: "+str(accu*100.0)+"%")



