import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
length=20000
gae_list=[1,0.99,0.98,0.96,0.92,0.84,0.68,0.36,0]
plt.ylim(ymin=-80,ymax=-30)
def plotf(folderlist,name):
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
    r1 = mean-stds
    r2 = mean+std
    plt.fill_between(np.array(list(range(len(mean)))), r1, r2,alpha=0.2)



def plotall(env='thrower',mode='bias'):
    if mode=='bias':
       flag='_bias_gae'
       flag1=' HVF'
    else:
       flag='_gae'
       flag1=' SVF'
    if env=='thrower':
        name='Thrower'+flag1
        path='thrower_5_0.15'
    else:
        name='Target Tracker'+flag1
        path='reach_4_0.05'
        
     
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.title(name)
    plt.grid()
    for gae in range(9):
        data=[]
        for index in range(3):
            data.append('result_setgae1/result_PPO'+path+flag+str(gae)+'_'+str(index))
        plotf(data,r'$\lambda$ = '+str(gae_list[gae]))


    #plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.legend(loc='lower right',ncol=3)

    plt.show()
    plt.savefig(env+mode+'.pdf')

plotall('thrower','nobias')


