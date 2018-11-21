import argparse
from time import time
from datetime import datetime
from prettytable import PrettyTable
import prettytable as pt
from os.path import isfile

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models.resnet import model_urls

###############################################################################
global rootdir, epoch_Num

rootdir = "/u/training/tra471/scratch/hw5"
epoch_Num = 50
output_name='summary.dat'
alpha = 0.001
checkpoint_folder ="/checkpoint"

checkpoint_folder = rootdir+checkpoint_folder
output_file = rootdir+'/'+output_name

################################################################################


parser = argparse.ArgumentParser(description='Deep Ranking Pytorch Implementation')

parser.add_argument('-F','--freeze',default=False,type=bool,metavar='T',
		    help=' Freeze some specific layers in pretrained ResNet')

parser.add_argument('-R','--checkpoint', default=False, type=bool, metavar='T',
                    help=' Recover from checkpoint or not (default: False)')
parser.add_argument('--recover-file',default='',type=str,metavar='PATH',
		    help="checkpoint name")

global args
args = parser.parse_args()
model_urls['resnet101'] = model_urls['resnet101'].replace('https://', 'http://')

#################################################################################

class TripletTrainDataset(Dataset):
    def __init__(self,rootdir,classfile,transform=None):
        self.rootdir = rootdir
        self.BaseSample, self.classes = self.readclassfile(classfile)
        self.transform =transform
        self.N = len(self.BaseSample)
        a,b,c = self.construct_dictionary()
        self.dics=a
        self.inverse_dics=b
        self.class_size=c
        self.total_class_num = len(self.classes)
    
    def readclassfile(self,classfile):
        fi = open(classfile,'r')
        lines =fi.readlines()
        fi.close()
        classes = ['name' for i in range(len(lines))]
        for i in range(len(lines)):
            classes[i] = lines[i].split()[0]
        classes.sort()
        samples = []
        for label, Class in enumerate(classes):
            imgfiles = Class+'_boxes.txt'
            fi = open(self.rootdir+'/'+Class+'/'+imgfiles)
            lines = fi.readlines()
            fi.close()
            for line in lines:
                imgName = line.split()[0]
                imgName = Class+'/images/'+imgName
                samples.append([imgName,label])
        return samples, classes
            
    
    
    def construct_dictionary(self):
        class_num = len(self.classes)
        dics = [[] for i in range(class_num)]
        inverse_dics= np.zeros(self.N)
        for i in range(self.N):
            label = self.BaseSample[i][1]
            dics[int(label)].append(i)
            inverse_dics[i] = len(dics[int(label)])-1
        class_sample_num = [len(dics[i]) for i in range(class_num)]
        
        return dics,inverse_dics,class_sample_num 
        
    def __len__(self):
        return len(self.BaseSample)
    
    def __getitem__(self,index):
        image,label = self.BaseSample[index]
        label = int(label)
        this_class_size = self.class_size[label]
        
        plus_inclass_index = np.random.randint(1,this_class_size-1)
        plus_inclass_index = (self.inverse_dics[index] + plus_inclass_index)%this_class_size
        plus_inclass_index = int(plus_inclass_index)
        plus_index = self.dics[label][plus_inclass_index]
        plus_image = self.BaseSample[plus_index][0]
        
        mins_class_index = np.random.randint(1,self.total_class_num-1)
        mins_class_index = (label+mins_class_index)%self.total_class_num
        mins_class_index = int(mins_class_index)
        mins_class_size = self.class_size[mins_class_index]
        mins_inclass_index = np.random.randint(mins_class_size)
        mins_index = self.dics[mins_class_index][mins_inclass_index]
        mins_image = self.BaseSample[mins_index][0]
        
        image = Image.open(self.rootdir+"/"+image).convert('RGB')
        plus_image = Image.open(self.rootdir+"/"+plus_image).convert('RGB')
        mins_image = Image.open(self.rootdir+"/"+mins_image).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
            plus_image = self.transform(plus_image)
            mins_image = self.transform(mins_image)
            
        return (image,plus_image,mins_image)


class TripletTestSet(Dataset):
    def __init__(self,folder,classnames,transform=None):
        
        self.root = folder
        self.classnames= classnames
        self.samples = self.openfile(folder+"/val_annotations.txt")
        self.transform = transform
        
    def openfile(self,filename):
        fi = open(filename,'r')
        samples = []
        for line in fi:
            line_str = line.split()
            img_name = line_str[0]
            img_label = line_str[1]
            img_class = self.classnames.index(img_label)
            samples.append([img_name,img_class])
        fi.close()
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        img_name = self.samples[index][0]
        img_label =self.samples[index][1]
        img = Image.open(self.root+"/images/"+img_name).convert('RGB')
        if self.transform:
            sample = self.transform(img)
        return sample,img_label

class BasicRes(nn.Module):
    def __init__(self):
        super(BasicRes,self).__init__()
        pretrained_res =models.resnet101(pretrained=True) 
        self.resnet = nn.Sequential(*list(pretrained_res.children())[:-1])
        self.fc1 = nn.Linear(2048, 4096)
    def forward(self,x):
        #out = F.interpolate(out,size=(224,224))
        out = F.interpolate(x,scale_factor=3.5)
        out = self.resnet(out)
        out = out.view(-1, 2048)
        out = self.fc1(out)
        return out
        
class RankNet(nn.Module):
    def __init__(self,basicres,embd_dim=4096):
        super(RankNet,self).__init__()
        self.Qnet = basicres()
        self.Pnet = basicres()
        self.Nnet = basicres()
    def forward(self,q,p,n):
        qem = self.Qnet(q)
        pem = self.Pnet(p)
        nem = self.Nnet(n)
        return (qem,pem,nem)
    
class TripletLossFunc(nn.Module):
    def __init__(self,g=1.0):
        super(TripletLossFunc,self).__init__()
        self.g = g
        
    def forward(self,q,p,n):
        matched = F.pairwise_distance(q,p)
        mismatched = F.pairwise_distance(q,n)
        loss = matched - mismatched+self.g
        loss = torch.clamp(loss,min=0.0)
        loss = torch.mean(loss)
        return loss
#####################################################################################
##  some functions ###########################
def Freeze(net):
    child_counter = 0
    print ('In ' + net.__class__.__name__ +':')
    for child in net.children():
        if child_counter in  [4,5] :
           for params in child.parameters():
               params.requires_grad = False
           print('\t child '+str(child_counter)+ ' is Frozen')
        print 
        if child_counter==6:
           sub_child_counter = 0
           sub_child_frozen=[]
           for sub_child in child.children():
               if sub_child_counter > 10 :
                  for params in sub_child.parameters():
                      params.requires_grad = False
                  sub_child_frozen.append(sub_child_counter)
           sub_child_counter +=1
        child_counter +=1

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    check_file = checkpoint_folder+'/'+filename
    count=1
    while isfile(check_file):
        filename = 'checkpoint'+str(count).zfill(2)
        filename = filename+'.pth.tar'
        check_file = checkpoint_folder + '/' + filename
        count +=1
    torch.save(state, check_file)


#####################################################################################
## Load Data ################################


print ("begin data load")
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_dir=rootdir+"/tiny-imagenet-200/train"
test_dir =rootdir+"/tiny-imagenet-200/val"
classfile =rootdir+"/tiny-imagenet-200/wnids.txt" 
#train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
#test_dataset = TripletTestSet(test_dir,train_dataset.classes,transform=transform)

trainset = TripletTrainDataset(train_dir, classfile,transform=transform)
testset = TripletTestSet(test_dir,trainset.classes,transform=transform)

trainloader = DataLoader(trainset, batch_size=6,shuffle=True, num_workers=16)

print ("finish data load")
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
################################################

## Initialization ###########################

print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ( "begin initialization")

Qnet = BasicRes().cuda()


if args.freeze:
    Freeze(Qnet)

criterion = TripletLossFunc()
optimizer = optim.SGD(Qnet.parameters(), lr=alpha, momentum=0.9 ,weight_decay=0.01)
#schedule = optim.lr_scheduler.ExponentialLR(optimizer,0.01)
decay = lambda ep:np.exp(-0.01*ep)
schedule = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=decay)

start_epoch = 0
average_epoch_time = 0.0

print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ( "finish initialization")


## Recover ##################################
if args.checkpoint:
    filename = "checkpoint.pth.tar"
    if args.recover_file:
        filename = args.recover_file
    filename = rootdir+'/'+filename
    if isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
       
        print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        average_epoch_time=checkpoint['epoch_time']
        Qnet.load_state_dict(checkpoint['QnetState'])
        #Pnet.load_state_dict(checkpoint['PnetState'])
        #Nnet.load_state_dict(checkpoint['NnetState'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
       
    print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))


## Train ####################################

print ("begin training")
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
epoch_cost = 0


epoch = start_epoch 
while epoch < epoch_Num:
    OUTPUT_FILE = open(output_file,'a')
    table = PrettyTable(['Epo', 'itr','curr_L','avg_L','total_L','fwT','bkT','avg_itr_T','avg_epo_T']\
                         ,float_format='2.2')
    table.hrules=3
    print (table)
    table.header=False
    tablerow_counter=0
    

    EPOCH_BEGIN_TIME = time()
    cycle =  len(trainloader)
    total_cost = 0
    schedule.step(epoch)
    for i,data in enumerate(trainloader,0):
        point1 = time()   #POINT 1, FORWARD BEGIN
        qimgs, pimgs, nimgs = data
        qimgs = torch.autograd.Variable(qimgs).cuda()
        pimgs = torch.autograd.Variable(pimgs).cuda()
        nimgs = torch.autograd.Variable(nimgs).cuda()
        
        optimizer.zero_grad()
        qemb = Qnet(qimgs)
        pemb = Qnet(pimgs)
        nemb = Qnet(nimgs)
        loss = criterion(qemb,pemb,nemb)
        cost= loss.item()        
        

        point2 = time()   #POINT 2, FORWARD END
        if cost !=0:
           loss.backward(retain_graph=False)
        optimizer.step()
        point3 = time()   #POINT 3, BACKWARD END
        
        total_cost += cost
        forward_time = point2 - point1
        backward_time = point3 - point2
        if (i+1)%64 ==0:
            avg_cost = total_cost/(i+1.0)
            avg_itr_time = (point3-EPOCH_BEGIN_TIME)/(i+1.)
            percentage = 1.*(i+1)/cycle
            average_epoch_time = (average_epoch_time*epoch+avg_itr_time*(i+1))/(epoch+percentage)
            table.add_row([str(epoch+1).zfill(2),str(i+1).zfill(5), "{:+.4E}".format(cost),\
                           "{:+.5E}".format(avg_cost), "{:+.5E}".format(total_cost), "{:.4f}".format(forward_time),\
                           "{:.4f}".format(backward_time), "{:.4f}".format(avg_itr_time), \
                           "{:.4f}".format(average_epoch_time/3600.0)])
            table.start=tablerow_counter
            print (table)
            tablerow_counter+=1
        del loss
        del qemb,pemb,nemb,qimgs,pimgs,nimgs
    EPOCH_END_TIME = time()
    average_epoch_time = ((EPOCH_END_TIME-EPOCH_BEGIN_TIME)+1.*epoch*average_epoch_time)/(epoch+1.)
    epoch_cost = total_cost/cycle
    del (table)
    lines = str(epoch)+'\t'+str(epoch_cost)+'\n'
    OUTPUT_FILE.write( lines  )
    OUTPUT_FILE.close()
        
    print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
    if (epoch+1) %1 ==0:
        print ("saving checkpoint ...")
        save_params={
            'epoch': epoch+1,
            'epoch_time':average_epoch_time,
            'QnetState':Qnet.state_dict(),
            #'PnetState':Pnet.state_dict(),
            #'NnetState':Nnet.state_dict(),
            'optimizer':optimizer.state_dict(),
        }
        save_checkpoint(save_params)
        print ('Done')
    print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
    if (epoch+1-start_epoch)%4 ==0:
        print ("trained " + str(epoch))
        print ("exit...")
        exit()
    epoch = epoch +1
########################################################################
