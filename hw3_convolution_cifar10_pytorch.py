
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from google.colab import files

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4,padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_drop = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(64, 64, 4,padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4,padding=2)
        self.conv5 = nn.Conv2d(64, 64,4,padding=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,3,padding=0)
        self.conv7 = nn.Conv2d(64,64,3,padding=0)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,64,3,padding=0)
        self.bn8 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,128)
        self.fc4 = nn.Linear(128,10)
        self.fc_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),negative_slope=0.1)
        x = F.leaky_relu(self.pool(self.conv2(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)),negative_slope=0.1)
        x = F.leaky_relu(self.pool(self.conv4(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.bn5(self.conv5(x)),negative_slope=0.1)
        x = F.leaky_relu(self.conv6(x),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.bn7(self.conv7(x)),negative_slope=0.1)
        x = F.leaky_relu(self.bn8(self.conv8(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,64*4*4)
        x = F.leaky_relu(self.fc1(x),negative_slope=0.1)
        x = F.leaky_relu(self.fc2(x),negative_slope=0.1)
        x = F.leaky_relu(self.fc3(x),negative_slope=0.1)
        x = self.fc4(x)
        return x

class Model_Trainer():
    def __init__(self,model,criterion,optimizer,batchSize,trainset,testset):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                            shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                            shuffle=False, num_workers=2)
        self.epoch_num_trained = 0
        self.costs = []
        self.costs_step = []
        self.test_accus = []
        self.train_accus =[]
    
    def train(self,epoch_Num):
        for epoch in range(epoch_Num):
            self.model.train()
            cost = 0
            cycle = len(self.trainloader)
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs = torch.autograd.Variable(inputs).cuda()
                labels = torch.autograd.Variable(labels).cuda()
                
                
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                cost += loss.item()/cycle
            self.epoch_num_trained +=1
            if (self.epoch_num_trained)%(2) == 0 or epoch==0:
                train_a =self.train_accu()
                test_a = self.test_accu()
                print(str(self.epoch_num_trained)+": loss, "+ str(cost)\
                     +", train_accu, "+ str(train_a)+ ", test_accu, "+ str(test_a))
                print (str(self.epoch_num_trained))
                print("train_accu: "+ str(train_a))
                print("test_accu: "+ str(test_a))
                self.test_accus.append(train_a)
                self.train_accus.append(test_a)
                self.costs.append(cost)
                self.costs_step.append(self.epoch_num_trained)
        #self.costs.append(cost)
        #self.costs_step.append(self.epoch_num_trained)
        return 
    
    def test_accu(self):
        
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = torch.autograd.Variable(images).cuda()
                labels = torch.autograd.Variable(labels).cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0*correct/total
    
    def train_accu(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.trainloader:
                images, labels = data
                images = torch.autograd.Variable(images).cuda()
                labels = torch.autograd.Variable(labels).cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0*correct/total

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

mynet = Net().cuda()
#mynet.load_state_dict(torch.load("model.par"))
criterion = nn.CrossEntropyLoss()
alpha = 0.001
optimizer = optim.Adam(mynet.parameters(),lr = alpha)
batch_size=128
mynet_trainer = Model_Trainer(mynet,criterion,optimizer,batch_size,trainset,testset)


mynet_trainer.train(50)

torch.save(mynet.state_dict(),"model.par" )

#mynet_trainer.train_accu()
#mynet_trainer.test_accu()


plt.plot(mynet_trainer.cost_step,mynet_trainer.test_accus,label="test accu")
plt.plot(mynet_trainer.cost_step,mynet_trainer.train_accus,label="train accu")
plt.legend(loc="best")
plt.show()

def Monte_Carlo_test(model_trainer,MC_size,images):
  model = model_trainer.model
  model.train()
  outs = np.zeros((MC_size,128,10))
  for i in range(MC_size):
    with torch.no_grad():
      images = torch.autograd.Variable(images).cuda()
      #labels = torch.autograd.Variable(labels).cuda()
      outputs = model(images)
      outputs = outputs.cpu().numpy()
      #print (outputs.shape)
      #print (outputs[1])
      expouts = np.exp(outputs)
      outputs = expouts/(np.sum(expouts,axis=1,keepdims=True))
      outs[i,:,:] = outputs[:,:]
  out = np.sum(outs,axis=0)/MC_size
  return out,outs
    
def Evaluate(model_trainer,MC_size,images):
  model = model_trainer.model
  model.eval()
  outs = np.zeros((128,10))
  with torch.no_grad():
    images, labels = data
    images = torch.autograd.Variable(images).cuda()
    #labels = torch.autograd.Variable(labels).cuda()
    outputs = model(images)
    outputs = outputs.cpu().numpy()
    expouts = np.exp(outputs)
    outs[:,:] = expouts/(np.sum(expouts,axis=1,keepdims=True))
  return outs

for i, data in enumerate(mynet_trainer.trainloader, 0):
  images, labels = data
  break
MCout,MCouts = Monte_Carlo_test(mynet_trainer,100,images)
Myout = Evaluate(mynet_trainer,1000,images)


diff = np.linalg.norm(Myout-MCout,ord=np.inf,axis=1)
accu = np.linalg.norm(MCout,ord=1,axis=1)
outs = np.linalg.norm(Myout,ord=1,axis=1)
accu_ratio = diff/accu


plt.figure()
plt.plot(accu_ratio,label="accu_ratio")
#plt.plot(accu,label="accu")
#plt.plot(outs,label="outs")
plt.legend()
#plt.savefig("mctest_ratio.png")
#files.download("mctest_ratio.png")


plt.figure()
accu = np.argmax(MCout,axis=1)
outs = np.argmax(Myout,axis=1)
plt.plot(accu-outs,label="accu")
#plt.plot(outs,label="outs")
plt.legend()
plt.savefig("mctest.png")
files.download("mctest.png")
print (1.*np.sum(accu==outs)/accu.shape[0])

