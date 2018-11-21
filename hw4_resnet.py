import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys 


class BasicBlock(nn.Module):
    def __init__(self,inchannel, outchannel,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block,num_blocks=[2,4,4,2], inchannel=32, num_classes=100):
        super(ResNet,self).__init__()
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(3,self.inchannel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(self.inchannel)
        self.dropout = nn.Dropout(p=0.5)
        self.layer1 = self._make_layers(block,num_blocks[0], 32,stride=1)
        self.layer2 = self._make_layers(block,num_blocks[1], 64,stride=2)
        self.layer3 = self._make_layers(block,num_blocks[2],128,stride=2)
        self.layer4 = self._make_layers(block,num_blocks[3],256,stride=2)
        #self.pool = nn.MaxPool2d(3,3)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(256, num_classes)
        
    def _make_layers(self,block, num_blocks,outchannel, stride=1):
        layers = [block(self.inchannel,outchannel,stride)]
        self.inchannel = outchannel
        for i in range(num_blocks-1):
            layers.append(block(self.inchannel,outchannel,stride=1))
            self.inchannel = outchannel
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out    

class Model_Trainer():
    def __init__(self,model,criterion,optimizer,batchSize,trainset,testset,scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                            shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                            shuffle=False, num_workers=2)
        self.epoch_num_trained = 0
        self.scheduler = scheduler
        self.costs = []
        self.costs_step = []
        self.test_accus = []
        self.train_accus =[] 
        
    def train(self,epoch_Num):
        for epoch in range(epoch_Num):
            self.model.train()
            self.scheduler.step(epoch)
            cost = 0
            cycle = len(self.trainloader)
            total_samples = 0
            total_correct = 0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs = torch.autograd.Variable(inputs).cuda()
                labels = torch.autograd.Variable(labels).cuda()
                
                
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                cost += loss.data/cycle
                if i %10 ==0 :
                  _, predicted = torch.max(outputs.data, 1)
                  total_correct += (predicted == labels).sum().item()
                  total_samples += labels.size(0)
            
            train_a = 100.0*total_correct/total_samples
            self.epoch_num_trained +=1
            if (self.epoch_num_trained)%(1) == 0 or epoch==0:
                test_a = self.test_accu()
                print(str(self.epoch_num_trained)+": loss, "+ str(cost) \
                      +", train_accu, "+ str(train_a) +", test_accu, "+ str(test_a) ) 
                self.costs.append(cost)
                self.test_accus.append(test_a)
                self.train_accus.append(train_a)
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
        

folder = str(sys.argv[1])
print ("root:  "+str(folder))
print 
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR100(root=folder+'/data', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR100(root=folder+'/data', train=False,
                                        download=False, transform=transform)

print ("data prepared")
mynet = ResNet(BasicBlock).cuda()
criterion = nn.CrossEntropyLoss()
alpha = 0.001
optimizer = optim.Adam(mynet.parameters(),lr = alpha)
batch_size=128
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,30,40],gamma=0.2)
mynet_trainer = Model_Trainer(mynet,criterion,optimizer,batch_size,trainset,testset,scheduler)

total_epoch = 50
print ("alpha ="+str(alpha) , "batch_size ="+str(batch_size), "total_epoch ="+str(total_epoch) )
print ("net setted up")

print 
print ("train... ")
mynet_trainer.train(total_epoch)

print 
print ("final test", mynet_trainer.test_accu())

torch.save(mynet.state_dict(),folder+"/model.par" )

exit()
