import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms

########################################################################################################################
def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get(seed=0,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,28,28]

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train1']=datasets.MNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test1']=datasets.MNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['train2']=datasets.MNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std), transforms.Lambda(lambda x: _permutate_image_pixels(x, np.random.permutation(28**2)))]))
    dat['test2']=datasets.MNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std), transforms.Lambda(lambda x: _permutate_image_pixels(x, np.random.permutation(28**2)))]))
    
    data[0]={}
    data[0]['name']='mnist-0-4'
    data[0]['ncla']=5
    data[1]={}
    data[1]['name']='mnist-5-9'
    data[1]['ncla']=5
    data[2]={}
    data[2]['name']='pmnist-0-4'
    data[2]['ncla']=5
    data[3]={}
    data[3]['name']='pmnist-5-9'
    data[3]['ncla']=5

    # TRAIN 1
    loader_train1=torch.utils.data.DataLoader(dat['train1'],batch_size=1,shuffle=False)
    data[0]['train1']={'x': [],'y': []}
    data[1]['train1']={'x': [],'y': []}

    # TEST 1
    loader_test1=torch.utils.data.DataLoader(dat['test1'],batch_size=1,shuffle=False)
    data[0]['test1']={'x': [],'y': []}
    data[1]['test1']={'x': [],'y': []}

    # TRAIN 2
    loader_train2=torch.utils.data.DataLoader(dat['train2'],batch_size=1,shuffle=False)
    data[2]['train2']={'x': [],'y': []}
    data[3]['train2']={'x': [],'y': []}

    # TEST 2
    loader_test2=torch.utils.data.DataLoader(dat['test2'],batch_size=1,shuffle=False)
    data[2]['test2']={'x': [],'y': []}
    data[3]['test2']={'x': [],'y': []}

    # FOR MNIST 0-4 AND MNIST 5-9 TRAIN SET
    for image,target in loader_train1:
        label = target.numpy()[0]
        if label<5:
            data[0]['train1']['x'].append(image)
            data[0]['train1']['y'].append(label)
        else:
            data[1]['train1']['x'].append(image)
            data[1]['train1']['y'].append(label-5)

    # FOR MNIST 0-4 AND MNIST 5-9 TEST SET
    for image,target in loader_test1:
        label = target.numpy()[0]
        if label<5:
            data[0]['test1']['x'].append(image)
            data[0]['test1']['y'].append(label)
        else:
            data[1]['test1']['x'].append(image)
            data[1]['test1']['y'].append(label-5)

    # FOR PERMUTATED MNIST 0-4 AND MNIST 5-9 TRAIN SET
    for image,target in loader_train2:
        label = target.numpy()[0]
        if label<5:
            data[2]['train2']['x'].append(image)
            data[2]['train2']['y'].append(label)
        else:
            data[3]['train2']['x'].append(image)
            data[3]['train2']['y'].append(label-5)
    
    # FOR PERMUTATED MNIST 0-4 AND MNIST 5-9 TRAIN SET
    for image,target in loader_test2:
        label = target.numpy()[0]
        if label<5:
            data[2]['test2']['x'].append(image)
            data[2]['test2']['y'].append(label)
        else:
            data[3]['test2']['x'].append(image)
            data[3]['test2']['y'].append(label-5)


    # "Unify" and save
    data[0]['train1']['x']=torch.stack(data[0]['train1']['x']).view(-1,size[0],size[1],size[2])
    data[0]['train1']['y']=torch.LongTensor(np.array(data[0]['train1']['y'],dtype=int)).view(-1)
    data[0]['test1']['x']=torch.stack(data[0]['test1']['x']).view(-1,size[0],size[1],size[2])
    data[0]['test1']['y']=torch.LongTensor(np.array(data[0]['test1']['y'],dtype=int)).view(-1)

    data[1]['train1']['x']=torch.stack(data[1]['train1']['x']).view(-1,size[0],size[1],size[2])
    data[1]['train1']['y']=torch.LongTensor(np.array(data[1]['train1']['y'],dtype=int)).view(-1)
    data[1]['test1']['x']=torch.stack(data[1]['test1']['x']).view(-1,size[0],size[1],size[2])
    data[1]['test1']['y']=torch.LongTensor(np.array(data[1]['test1']['y'],dtype=int)).view(-1)

    data[2]['train2']['x']=torch.stack(data[2]['train2']['x']).view(-1,size[0],size[1],size[2])
    data[2]['train2']['y']=torch.LongTensor(np.array(data[2]['train2']['y'],dtype=int)).view(-1)
    data[2]['test2']['x']=torch.stack(data[2]['test2']['x']).view(-1,size[0],size[1],size[2])
    data[2]['test2']['y']=torch.LongTensor(np.array(data[2]['test2']['y'],dtype=int)).view(-1)

    data[3]['train2']['x']=torch.stack(data[3]['train2']['x']).view(-1,size[0],size[1],size[2])
    data[3]['train2']['y']=torch.LongTensor(np.array(data[3]['train2']['y'],dtype=int)).view(-1)
    data[3]['test2']['x']=torch.stack(data[3]['test2']['x']).view(-1,size[0],size[1],size[2])
    data[3]['test2']['y']=torch.LongTensor(np.array(data[3]['test2']['y'],dtype=int)).view(-1)
    
    # Validation
    data[0]['valid'] = {}
    data[0]['valid']['x'] = data[0]['train1']['x'].clone()
    data[0]['valid']['y'] = data[0]['train1']['y'].clone()

    data[1]['valid'] = {}
    data[1]['valid']['x'] = data[1]['train1']['x'].clone()
    data[1]['valid']['y'] = data[1]['train1']['y'].clone()

    data[2]['valid'] = {}
    data[2]['valid']['x'] = data[2]['train2']['x'].clone()
    data[2]['valid']['y'] = data[2]['train2']['y'].clone()

    data[3]['valid'] = {}
    data[3]['valid']['x'] = data[3]['train2']['x'].clone()
    data[3]['valid']['y'] = data[3]['train2']['y'].clone()


    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################

