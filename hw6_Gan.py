import argparse
from time import time
from datetime import datetime
from os.path import isfile

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#################################################################################

global rootdir, batch_size, n_class, n_z,checkpoint_dir,checkpoint_name,gen_train

rootdir = '/u/training/tra471/scratch/hw6'
batch_size = 128
n_classes = 10
n_z = 100
image_dir = 'image'
checkpoint_dir='checkpoint'
checkpoint_name='checkpoint'
wall_time = [11,30,0]
num_epochs = 250
gen_train = 1

image_dir = rootdir+'/'+image_dir
checkpoint_dir = rootdir+'/'+checkpoint_dir
wall_time = wall_time[0]*3600.0+ wall_time[1]*60.0 + wall_time[2] 
#################################################################################

parser = argparse.ArgumentParser(description='Deep Ranking Pytorch Implementation')


parser.add_argument('-F','--recover-file',default='',type=str,metavar='PATH',
		    help="checkpoint name")

global args
args = parser.parse_args()


## Models ###############################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, 3,padding=1)
        self.ln1 = nn.LayerNorm([196,32,32])
        self.conv2 = nn.Conv2d(196, 196, 3,padding=1,stride=2)
        self.ln2 = nn.LayerNorm([196,16,16])
        self.conv3 = nn.Conv2d(196, 196, 3,padding=1)
        self.ln3 = nn.LayerNorm([196,16,16])
        self.conv4 = nn.Conv2d(196, 196, 3,padding=1,stride=2)
        self.ln4 = nn.LayerNorm([196,8,8])
        self.conv5 = nn.Conv2d(196, 196,3,padding=1)
        self.ln5 = nn.LayerNorm([196,8,8])
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.ln6 = nn.LayerNorm([196,8,8])
        self.conv7 = nn.Conv2d(196,196,3,padding=1)
        self.ln7 = nn.LayerNorm([196,8,8])
        self.conv8 = nn.Conv2d(196,196,3,padding=1,stride=2)
        self.ln8 = nn.LayerNorm([196,4,4])
        self.maxpool = nn.MaxPool2d(4,4)

        self.conv_drop = nn.Dropout(p=0.25)
     
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.conv1(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln2(self.conv2(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln3(self.conv3(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln4(self.conv4(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln5(self.conv5(x)),negative_slope=0.1)
        x = F.leaky_relu(self.ln6(self.conv6(x)),negative_slope=0.1)
        x = self.conv_drop(x)
        x = F.leaky_relu(self.ln7(self.conv7(x)),negative_slope=0.1)
        x = F.leaky_relu(self.maxpool( self.ln8(self.conv8(x)) ),negative_slope=0.1)
        
        x = x.view(-1,196)
        score = self.fc1(x)
        label = self.fc10(x)
        return score,label

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100,196*4*4)
        self.conv1 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn1 = nn.BatchNorm2d(196)
        self.conv2 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(196)
        self.conv3 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn3 = nn.BatchNorm2d(196)
        self.conv4 = nn.Conv2d(196, 196, 3,padding=1,stride=1)
        self.bn4 = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196,196,3,padding=1)
        self.bn6 = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196,196,4,padding=1,stride=2)      
        self.bn7 = nn.BatchNorm2d(196)
        self.conv8 = nn.Conv2d(196,3,3,padding=1,stride=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = x.view(-1,196,4,4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.tanh(self.conv8(x))
        
        return x

## printing tools ##################################################################################

class PrintTable():
    def __init__(self,header,Format):
        self.header = [' '+head+' ' for head in header]
        self.format = Format
        self.field_width = [len(head) for head in self.header]
        self.fieldN = len(header)
        self.vrule = '|'
        self.hrule = '-'
        self.cross = '+'
        self.withhead = True
        self.with_bottom_rule = True  
    def _get_header_str(self):
        headerstr=''+self.vrule
        myheader = []
        for i in range(len(self.header)) :
            remains = self.field_width[i] - len(self.header[i])
            left_space = int(remains/2)
            right_space = int(remains - left_space)
            myheader.append( ' '*left_space+self.header[i] + ' '*right_space)
        for i in range(len(self.header)):
            headerstr+=str(myheader[i])+'|'
        return headerstr

    def _get_rule(self):
        rules = ''+self.cross
        for i in range(self.fieldN):
            rules +=self.hrule*self.field_width[i] +self.cross
        return rules


    def printrow(self,row):
        if len(row) != self.fieldN:
            print ("wrong length")
            return 

        myrow = []
        for i in range(len(row)):
            myformat = self.format[i]
            item = row[i]
            if myformat=='':
               myrow.append(' '+ str(item) + ' ')
            else:
               myrow.append(' '+myformat.format(item)+' ')
        for i in range(len(myrow)):
            self.field_width[i] = max(self.field_width[i],len(myrow[i]))
            remains = self.field_width[i] - len(myrow[i])
            left_space = int(remains/2)
            right_space = int(remains - left_space)
            myrow[i] = ' '*left_space+myrow[i] + ' '*right_space
        myrow_str=''+self.vrule
        for i in range(len(myrow)):
            myrow_str+= str(myrow[i]) +self.vrule
        rules = self._get_rule()
        if self.withhead:
            print(rules)
            print( self._get_header_str())
            print(rules)
        print(myrow_str)
        if self.with_bottom_rule:
            print(rules)
        return

    def printrule(self):
        print(self._get_rule())

## some functions ###################################################################################
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def time_format(time_in_seconds):
    hours, rem = divmod(time_in_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    seconds = int(round(seconds))
    time_str = "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),seconds)
    return time_str

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

def save_checkpoint(state, filename=checkpoint_name):
    check_file = checkpoint_dir+'/'+filename+'.pth.tar'
    count=1
    while isfile(check_file):
        filename = 'checkpoint'+str(count).zfill(2)
        filename = filename+'.pth.tar'
        check_file = checkpoint_dir + '/' + filename
        count +=1
    torch.save(state, check_file)
    print ("save checkpoint to " + str(check_file))

def get_accu(model,testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
         for data in testloader:
             images, labels = data
             images = torch.autograd.Variable(images).cuda()
             labels = torch.autograd.Variable(labels).cuda()
             _, outputs = model(images)
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
    return 100.0*correct/total


## Load Data ##################################################################################
 
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ("Loading Data")  
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root=rootdir+'/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=rootdir+'/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
print ("Done")



### Initialization ##############################################################################
print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ("Initialization")


aD =  Discriminator()
aD.cuda()

aG = Generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()
epoch_start = 0


np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

print ("Done")
## Load checkpoint ###############################################################################
if args.recover_file:
    filename = args.recover_file
    filename = rootdir+'/'+filename
    if isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
       
        print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        checkpoint = torch.load(filename)
        epoch_start = checkpoint['epoch']
        aD.load_state_dict(checkpoint['DisState'])
        aG.load_state_dict(checkpoint['GenState'])
        optimizer_g.load_state_dict(checkpoint['optimizerg'])
        optimizer_d.load_state_dict(checkpoint['optimizerd'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
       
    print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))


## train #########################################################################################

print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
print ('train')
START = time()
epoch = epoch_start
while epoch < num_epochs:
    EPOCH_START = time()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0
    total_loss5 = 0.0
    total_accu1 = 0.0
    total_loss6 = 0.0
    total_gnumb = 0.0
    print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
    print 
    print ('++++++++++++++++++++++++++++++++')
    print ('++++        Epoch '+str(epoch).zfill(3)+'       ++++')
    print ('++++++++++++++++++++++++++++++++')
    print 
    #print ('+-------+-----+---------------+-------------+-------------+---------------+--------------+-----------+--------+------------+-----------+')
    #print ('| epoch | itr | gd_penal_loss | D_fake_loss | D_real_loss | D_rlabel_loss | D_flabel_loss| D_tr_accu | G_loss | epoch_time | total_time| ')
    #print ('+-------+-----+---------------+-------------+-------------+---------------+--------------+-----------+--------+------------+-----------+')
    table_head = ['epoch','itr','gd_penal_loss','D_fake_loss','D_real_loss','D_rlabel_loss','D_flabel_loss','D_tr_accu','G_loss','epoch_time','total_time']
    table_format = ['','','{:+.3E}','{:+.3E}','{:+.3E}','{:+.3E}','{:+.3E}','','{:+.3E}','','']
    table = PrintTable(table_head,table_format)
    table.with_bottom_rule = False
    aG.train()
    aD.train()
    cycle = len(trainloader)
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
        for group in optimizer_g.param_groups:
             for p in group['params']:
                  state = optimizer_g.state[p]
                  if('step' in state and state['step']>=1024):
                      state['step'] = 1000
        for group in optimizer_d.param_groups:
             for p in group['params']:
                  state = optimizer_d.state[p]
                  if('step' in state and state['step']>=1024):
                      state['step'] = 1000



        # Train Generator
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()
            
            total_loss6 = gen_cost.item()
            total_gnumb += 1.0
            optimizer_g.step()
            
        # Train Discriminator
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)
        
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)
        
        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()
        
        disc_real_source, disc_real_class = aD(real_data)
        
        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
        
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)
        
        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
        
        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()
        
        optimizer_d.step()
        
        total_loss1 += gradient_penalty.item()
        total_loss2 += disc_fake_source.item()
        total_loss3 += disc_real_source.item()
        total_loss4 += disc_real_class.item()
        total_loss5 += disc_fake_class.item()
        total_accu1 += accuracy

        if ((batch_idx%50)==0):
            aveg_loss1 = total_loss1/(batch_idx+1)
            aveg_loss2 = total_loss2/(batch_idx+1)
            aveg_loss3 = total_loss3/(batch_idx+1)
            aveg_loss4 = total_loss4/(batch_idx+1)
            aveg_loss5 = total_loss5/(batch_idx+1)
            aveg_accu1 = total_accu1/(batch_idx+1)
            aveg_loss6 = total_loss6/(total_gnumb)
            #out_str  = str(epoch).zfill(3)+' '
            #out_str += str(batch_idx).zfill(3)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss1)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss2)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss3)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss4)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss5)+ ' '
            #out_str +=  "{:.3f}".format(aveg_accu1)+ ' '
            #out_str += "{:+.3E}".format(aveg_loss6)+ ' '
            percentage = 1.*(batch_idx+1)/cycle
            total_time = time()-START
            average_time = (total_time)/(epoch-epoch_start+percentage)
            #out_str += time_format(average_time)+ ' '
            #out_str += time_format(total_time)
            #print (out_str)
            myrow = [str(epoch).zfill(3), str(batch_idx).zfill(3), aveg_loss1,\
                     aveg_loss2, aveg_loss3, aveg_loss4, aveg_loss5, '{:.2f}'.format(aveg_accu1)+'%',\
                     aveg_loss6, time_format(average_time),time_format(total_time)]
            table.printrow(myrow)
            if table.withhead:
                table.withhead=False
    table.printrule()
    del table

    test_accu = get_accu(aD,testloader)
    OUTPUTFILE = open(rootdir+'/summary.dat','a')
    OUTPUTFILE.write(str(epoch) + '\t' + '{:.3f}'.format(test_accu)+ '\t' + '{:.3f}'.format(aveg_accu1) +'\n')
    OUTPUTFILE.close()
    print ("test accuracy: \t " + '{:.3f}'.format(test_accu)+'%')
    print ("train accuracy:\t " + '{:.3f}'.format(aveg_accu1)+'%') 

    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plot_name = image_dir+'/%s.png' % str(epoch).zfill(3)
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)
    print ("plot saved to : " + str(plot_name) )
    time_remain =  wall_time-(time()-START)
    could_not_continue = ( time_remain < (1.1*average_time) ) 


    if (epoch+1)%5 ==0 or could_not_continue:
        print ("saving checkpoint ...")
        save_params={
            'epoch': epoch+1,
            'DisState':aD.state_dict(),
            'GenState':aG.state_dict(),
            'optimizerd':optimizer_d.state_dict(),
            'optimizerg':optimizer_g.state_dict(),
        }
        save_checkpoint(save_params)
    print ('epoch takes time : ' + time_format(time()-EPOCH_START))
    print ('Finished epoch '+ str(epoch))
    if could_not_continue:
        print ( datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        print ("remaining: " + time_format(time_remain))
        print ("could not continue")
        print ("exit...")
        exit()
    epoch +=1

torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')
