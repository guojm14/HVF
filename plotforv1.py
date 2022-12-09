import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
length=20000
def plotc(folderlist,name):
    
    line=[]
    for folder in folderlist:
        data=open(folder+'/closs_.txt').readlines()
        a=np.array(list(map(float,data)))
        a=a[:length//16]
        line.append(np.convolve(a, np.ones(((200//16),))/(200//16), mode='valid').reshape((1,-1)))
    temp=np.concatenate(line)
    mean=np.mean(temp,0)
    std=np.std(temp,0)
         
    plt.plot(np.array(list(range(len(mean))))*16,mean,label=name)
    #plt.plot(np.array(list(range(len(mean)))),std,label=name)
    r1 = mean-std
    r2 = mean+std
    plt.fill_between(np.array(list(range(len(mean)))), r1, r2,alpha=0.2)

def plotr(folderlist,name):
    line=[]
    for folder in folderlist:
        data=open(folder+'/t.txt').readlines()
        a=np.array(list(map(float,data)))
        a=a[:length]
        line.append(np.convolve(a, np.ones((200,))/200, mode='valid').reshape((1,-1)))
    temp=np.concatenate(line)
    mean=np.mean(temp,0)
    std=np.std(temp,0)

    plt.plot(np.array(list(range(len(mean)))),mean,label=name)
    #plt.plot(np.array(list(range(len(mean)))),std,label=name)
    r1 = mean-std
    r2 = mean+std
    plt.fill_between(np.array(list(range(len(mean)))), r1, r2,alpha=0.2)



def plotall(env='thrower',mode='reward'):
    if mode=='reward':
       plotf=plotr
    else:
       plotf=plotc
    if env=='thrower':
        name='Thrower'
        path='thrower'
    else:
        name='Target Tracker'
        path='reach'
        
    
    plt.xlabel('episodes')
    plt.ylabel(mode)
    plt.title(name)
    plt.grid()
    gae=[0,0,5,6]
    data=[]
    for index in range(3):
        data.append('result_setgae1/result_PPO'+path+'_gae'+str(gae[0])+'_'+str(index))
    plotf(data,'SVF')
    data=[]
    for index in range(3):
        data.append('result_setgae1/result_PPO'+path+'_bias_gae'+str(gae[1])+'_'+str(index))
    plotf(data,'HVF')
    data=[]
    for index in range(3):
        data.append('result_setgae1/result_PPO'+path+'_gae'+str(gae[2])+'_'+str(index))
    plotf(data,r'SVF-gae($\lambda=0.84$)')
    data=[]
    for index in range(3):
        data.append('result_setgae1/result_PPO'+path+'_bias_gae'+str(gae[3])+'_'+str(index))
    plotf(data,r'HVF-gae($\lambda=0.68$)')



    plt.legend(loc='lower right')

    plt.show()
    plt.savefig(env+mode+'.pdf')

plotall('reach','reward')

