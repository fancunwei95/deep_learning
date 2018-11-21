class Network:
    
    def __init__(self,shape_array, xtrain, ytrain):
        # shape_array is a 1D array each term means the node number of each layer
        # please include the input layer and output layer as well

        global np
        import numpy as np 
        self.layer_shape =np.copy(shape_array)
        self.layer_num = shape_array.shape[0]
        self.W_matrix_dict ={i:0.1*(np.random.randn(shape_array[i],shape_array[i+1])) \
                             for i in range(self.layer_num-1)} 
        self.b_matrix_dict = {i:0.1*np.random.randn(1,shape_array[i+1])\
                      for i in range(self.layer_num-1)}
        
        self.dW_dict = {i:np.zeros((shape_array[i],shape_array[i+1]))\
                       for i in range(self.layer_num-1)}
        self.db_dict = {i:np.zeros((1,shape_array[i+1]))\
                       for i in range(self.layer_num-1)}
        
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.costs=[]
        self.epoch_num_trained = 0

        
    def activation(self,x):
        #return np.exp(x)/(1.0+np.exp(x))
        return np.where(x>0,x,0)

    def activation_der(self,y):
        #return y*(1-y)
        return np.where(y>0,1,0)
        
    def predict(self,xbatch,activate):
        # xbatch each row is a sample
        N = self.layer_num
        #print (xbatch.shape[1],self.layer_shape[0])
        assert (xbatch.shape[1] == self.layer_shape[0])
        self.layer_value = {i:np.zeros((xbatch.shape[0],self.layer_shape[i]))\
                           for i in range(self.layer_num)}
        self.layer_value[0][:,:] = xbatch[:,:]
        for i in range(N-2):
            W = self.W_matrix_dict[i]
            b = self.b_matrix_dict[i]
            self.layer_value[i+1] = activate(self.layer_value[i].dot(W)+b)
        W = self.W_matrix_dict[N-2]
        b = self.b_matrix_dict[N-2]
        final_layer = np.exp(self.layer_value[N-2].dot(W)+b)
        #final_layer = self.layer_value[N-2].dot(W)+b
        final_layer = final_layer/(np.sum(final_layer,axis=1,keepdims=True))
        self.layer_value[N-1] = final_layer
        return final_layer
    
    def cost_func(self,xbatch,ybatch,activate):
        predict = self.predict(xbatch,activate)
        predict = np.log(predict)
        cost = -1.0*np.sum(predict*ybatch)/xbatch.shape[0]
        return cost
    
    def back_propogation(self,ybatch,activation_der):
        # last layer:
        N = self.layer_num
        dY = self.layer_value[N-1] - ybatch
        self.dW_dict[N-2] = (self.layer_value[N-2].T).dot(dY)
        self.db_dict[N-2] = np.sum(dY,axis=0,keepdims=True)
        for j in range(N-2):
            i = N-3-j
            dY = dY.dot(self.W_matrix_dict[i+1].T)*activation_der(self.layer_value[i+1])
            self.dW_dict[i] = (self.layer_value[i].T).dot(dY)
            self.db_dict[i] = np.sum(dY,axis=0,keepdims=True)
        return 
    
    def mini_batch_train(self,batch_size,alpha = 0.001,stepNum=1000):
        train_data_size = self.xtrain.shape[0]
        batch_size = batch_size
        shuffle_index = np.arange(train_data_size)
        
        stratify_index = [int(i*batch_size) for i in range(int(np.floor(train_data_size/batch_size)) )]
        stratify_index.append(train_data_size)
        cycle = len(stratify_index)-1
        N = self.layer_num

        for epoch in range(stepNum):
            cost = 0.0
            np.random.shuffle(shuffle_index)
            shuffledX = self.xtrain[shuffle_index,:]
            shuffledY = self.ytrain[shuffle_index,:]
            
            for i in range(cycle):
                index = np.arange(stratify_index[i%cycle],stratify_index[i%cycle+1])
                xbatch = shuffledX[index,:]
                ybatch = shuffledY[index,:]
                loss = self.cost_func(xbatch,ybatch,self.activation)
                self.back_propogation(ybatch,self.activation_der)
                self.W_matrix_dict = {i:self.W_matrix_dict[i]-alpha*self.dW_dict[i] for i in range(N-1)}
                self.b_matrix_dict = {i:self.b_matrix_dict[i]-alpha*self.db_dict[i] for i in range(N-1)}
                cost += loss/cycle
            self.epoch_num_trained +=1
            self.costs.append(cost)
            if (epoch)%(10) == 0 or epoch ==0:
                #print (epoch)
                print(str(epoch+1)+": loss, "+ str(cost))
                #self.costs.append(cost)
            
        return 1

if __name__=="__main__":
	import matplotlib.pyplot as plt 
	import h5py
	import time
	import numpy as np
        
	f = h5py.File('MNISTdata.hdf5','r')
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
	dimx = xtrain.shape[1]
	netshape=np.array([dimx,60,10])
	mynet = Network(netshape,xtrain,ytrain)

	start = time.time()
	mynet.mini_batch_train(60,stepNum=50)
	dura = time.time()-start
	print ("time: "+str(dura) +" s" )

	mypredict_test = mynet.predict(xtest,mynet.activation)
	mypredict_test = np.argmax(mypredict_test,axis=1)
	accu = np.sum(1.*(mypredict_test == ytest.flatten()) )/mypredict_test.shape[0]
	print ("test accuracy: "+str(accu*100.0)+"%")

	mypredict_train = mynet.predict(xtrain,mynet.activation)
	mypredict_train = np.argmax(mypredict_train,axis=1)
	ytrain_ori = go_back_from_one_hot(ytrain)

	accu = np.sum(1.*(mypredict_train == ytrain_ori.flatten()) )/mypredict_train.shape[0]
	print ("train accuracy: "+str(accu*100.0)+"%")
	
	columns = 5
	rows = 5
	fig=plt.figure(figsize=(10, 10))
	for i in range(1, columns*rows +1):
    	    index = i+np.random.randint(0,8000)
    	    pixels = np.array(xtest[index]).reshape(28,28)
    	    ax = fig.add_subplot(rows, columns, i)
    	    ax.imshow(pixels, cmap='gray',interpolation="spline16")
    	    ax.set_title("read as "+str(mypredict_test[index]))
    	    ax.axis('off')
	plt.show()











