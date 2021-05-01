import sys,os,argparse,time
import numpy as np
import torch

import utils

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=50,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
args=parser.parse_args()

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment

from dataloaders import mnist2 as dataloader

# Args -- Approach

from approaches import hat as approach


# Args -- Network

from networks import mlp_hat as network


########################################################################################################################

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)

# Initssn
print('Inits...')
net=network.Net(inputsize,taskcla).cuda()
utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)
models = []
# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    # Get data
    if t == 0 or t == 1:
        xtrain=data[t]['train1']['x'].cuda()
        ytrain=data[t]['train1']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t
    else:
        xtrain=data[t]['train2']['x'].cuda()
        ytrain=data[t]['train2']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*100)

    # Test

    for u in range(t+1):
        print('t == {}, u == {}'.format(t,u))
        if u == 0 or u==1:
            xtest=data[u]['test1']['x'].cuda()
            ytest=data[u]['test1']['y'].cuda()
            test_loss,test_acc=appr.eval(u,xtest,ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc
            lss[t,u]=test_loss
        else:
            xtest=data[u]['test2']['x'].cuda()
            ytest=data[u]['test2']['y'].cuda()
            test_loss,test_acc=appr.eval(u,xtest,ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc
            lss[t,u]=test_loss
    

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
