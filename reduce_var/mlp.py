import torch
import math
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal as multinormal

def identity(x):
    return x


class FC(nn.Module):
    def __init__(self, inc, outc, act='id', norm='id'):
        super(FC, self).__init__()
        if act == 'relu':
            self.act = F.relu
        elif act == 'lrelu':
            self.act = F.leaky_relu
        elif act == 'id':
            self.act = identity
        else:
            print('error activation function')

        if norm == 'bn':
            self.norm = nn.BatchNorm1d(outc)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(outc)
        elif norm == 'id':
            self.norm = identity
        else:
            print('error normalization')
        self.fc = nn.Linear(inc, outc)

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))




class Actor(nn.Module):
    def __init__(self, in_dim, h_dim,out_dim,layer_num):
        super(Actor, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.layer_num = layer_num

        self.fc = []
        self.fc.append(FC(self.in_dim, self.h_dim, act='lrelu'))
        for i in range(self.layer_num-1):
            self.fc.append(FC(self.h_dim, self.h_dim, act='lrelu'))
        self.fc = nn.Sequential(*self.fc)

        self.outfc_mu=FC(self.h_dim, out_dim)
        self.outfc_sigma=FC(self.h_dim, out_dim)
    def forward(self, x):
        h=self.fc(x)
        #mu = torch.tanh(self.outfc_mu(h))
        mu=self.outfc_mu(h)
        sigma = torch.clamp(self.outfc_sigma(h), min=-20, max=2).exp()
        return (mu,sigma)

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim,out_dim,layer_num):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.layer_num = layer_num

        self.fc = []
        self.fc.append(FC(self.in_dim, self.h_dim, act='lrelu'))
        for i in range(self.layer_num-1):
            self.fc.append(FC(self.h_dim, self.h_dim, act='lrelu'))
        self.fc = nn.Sequential(*self.fc)

        self.outfc=FC(self.h_dim, out_dim)
    def forward(self, x):
        #print(x)
        out = self.outfc(self.fc(x))
        return out


class CLUB(nn.Module): 

    def __init__(self, x_dim, hidden_size,y_dim):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ELU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ELU(),
                                  nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ELU(),
                                  nn.Linear(hidden_size//2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ELU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ELU(),
                                      nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ELU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def mi_est(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)


class Pre(nn.Module):
    def __init__(self, input_dim,action_dim,h_dim=32,l_dim=16):
        super(Pre, self).__init__()
        self.state_pre = MLP(in_dim=input_dim+action_dim+l_dim, h_dim=h_dim, out_dim = input_dim, layer_num=2)
        
        self.pre_base = MLP(in_dim=input_dim+action_dim,h_dim=h_dim, out_dim = input_dim, layer_num=2)

        self.lucky_net = MLP(in_dim=input_dim+action_dim+input_dim, h_dim=h_dim, out_dim = l_dim, layer_num=2)

    def get_l(self,x):
        return F.tanh(self.lucky_net(x))
    def forward(self,state,action,state_next):
        l=F.tanh(self.lucky_net(torch.cat([state,action,state_next],1)))
        return self.pre_base(torch.cat([state,action],1))+self.state_pre(torch.cat([state,action,l],1)),l


class Pre_1(nn.Module):
    def __init__(self, input_dim,action_dim,h_dim=32,l_dim=16):
        super(Pre_1, self).__init__()
        self.state_pre = MLP(in_dim=input_dim+action_dim+l_dim, h_dim=h_dim, out_dim = input_dim, layer_num=2)
                
        self.lucky_net = MLP(in_dim=input_dim+action_dim+input_dim, h_dim=h_dim, out_dim = l_dim, layer_num=2)

    def get_l(self,x):
        return F.tanh(self.lucky_net(x))
    def forward(self,state,action,state_next):
        l=F.tanh(self.lucky_net(torch.cat([state,action,state_next],1)))
        return self.state_pre(torch.cat([state,action,l],1)),l


class Critic_net_lstm(nn.Module):
    def __init__(self, input_dim,h_dim,l_dim):
        super(Critic_net_lstm, self).__init__()
        self.prel=nn.Linear(l_dim,h_dim//4)
        self.lnet=nn.GRU(h_dim//4, h_dim//4)#nn.LSTM(h_dim//4, h_dim//4)
        self.fc=nn.Linear(h_dim//4,h_dim//4)
        self.snet=MLP(input_dim,h_dim,h_dim,4)
        self.out=MLP(h_dim+h_dim//4,h_dim,1,2)

    def forward(self,state,l,hidden):
        l=F.elu(self.prel(l))
        l_feature,_=self.lnet(l.flip(0),hidden[0])
        l_feature=F.elu(self.fc(l_feature.flip(0).transpose(0,1)))
        s_feature=self.snet(state)
        #return self.out(s_feature)
        return self.out(torch.cat([l_feature,s_feature],2))


class Critic(nn.Module):
    def __init__(self, input_dim,h_dim,l_dim):
        super(Critic, self).__init__()
        #self.lnet=nn.LSTM(l_dim, h_dim)

        self.snet=MLP(input_dim,h_dim,h_dim,4)
        self.out=MLP(h_dim,h_dim,1,2)

    def forward(self,state):
        
        #l_feature,_=self.lnet(l,hidden)
        #l_feature=l_feature.squeeze().flip(0)
        s_feature=self.snet(state)

        return self.out(s_feature)







class Lucky(nn.Module):
    def __init__(self, input_dim,action_dim,h_dim=64,l_dim=32):
        super(Lucky, self).__init__()
        # action mean range -1 to 1

        self.anet = Actor(
            in_dim=input_dim, h_dim=h_dim,out_dim=action_dim,layer_num=4)

        self.cnet_l = Critic_net_lstm(input_dim=input_dim,h_dim=h_dim,l_dim=l_dim)


        self.cnet = Critic(input_dim=input_dim,h_dim=h_dim,l_dim=l_dim)

        self.lnet = Pre(input_dim=input_dim,action_dim=action_dim,h_dim=h_dim,l_dim=l_dim)

        self.qnet=CLUB(input_dim+action_dim,h_dim,l_dim)

    def actor(self, state):
        out = self.anet(state)
        return out

    def critic(self, state):
        out = self.cnet(state)
        return out

    def forward(self):
        raise NotImplementedError

    def value(self, state):
        out = self.critic(state)
        return out



