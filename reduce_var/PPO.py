import torch
import torch.nn as nn
#from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
#from .trans_coma import AC_coma
from .mlp import ActorCritic as AC
import numpy as np
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy
import torch.nn.functional as F
from itertools import chain
import random
from reduce_var.mlp import Lucky
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.nextstates=[]
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.entropy = []
        self.values =[]

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.entropy[:]
        del self.values[:]
        del self.nextstates[:]


class Buffer:
    def __init__(self):
        self.actions = []
        self.states=[]
        self.nextstates = []
        self.rewards = []
    def add(self,state,action,reward,nextstate):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.nextstates.append(nextstate)

    def pop(self):
        del self.actions[0]
        del self.states[0]
        del self.rewards[0]
        del self.nextstates[0]
    def __len__(self):
        return len(self.actions)

class PPO:
    def __init__(self, action_dim, input_dim,lr, betas, gamma, gpu, h_dim=64,l_dim=16,log_folder=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.l_dim = l_dim
        self.action_dim=action_dim
        self.input_dim=input_dim
        self.policy = Lucky(input_dim,action_dim, h_dim=h_dim,l_dim=l_dim).cuda(gpu)

        self.opt_a = torch.optim.Adam(
            self.policy.anet.parameters(), lr=lr, betas=betas,weight_decay=0.0000)

        self.opt_q=torch.optim.Adam(
            self.policy.qnet.parameters(), lr=lr, betas=betas,weight_decay=0.0000)

        self.opt_l=torch.optim.Adam(
            self.policy.lnet.parameters(), lr=lr, betas=betas,weight_decay=0.0000)
        
        self.opt_c=torch.optim.Adam(
            self.policy.cnet.parameters(), lr=lr, betas=betas,weight_decay=0.0000)        

        self.opt_c_l=torch.optim.Adam(
            self.policy.cnet_l.parameters(), lr=lr, betas=betas,weight_decay=0.0000)

        self.MseLoss = nn.MSELoss()
        self.gpu = gpu
       
        self.logdir = log_folder
        
        self.buffer=Buffer()
        self.buffer_size=20000
        self.batch_size=256
        #self.l_zero=torch.zeros((1,1,l_dim)).cuda(self.gpu)

        self.hxzero=torch.zeros((1,16,64)).cuda(self.gpu)
        self.cxzero=torch.zeros((1,16,64)).cuda(self.gpu)

    def select_action(self, state, memory):
        
        return self.act(state.cuda(self.gpu), memory).cpu()


    def act(self, state, memory):
        mean,std = self.policy.actor(state)
        cov_mat = torch.diag(std).cuda(self.gpu)
        dist = MultivariateNormal(mean,cov_mat)
        action = dist.sample()
        memory.states.append(state)
        memory.logprobs.append(dist.log_prob(action))
        memory.actions.append(action)
        return action

    def evaluate(self, state, action):
        mean,std = self.policy.actor(state)
        cov_mat = torch.diag_embed(std).cuda(self.gpu)
        dist = MultivariateNormal(mean,cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy



        
    def update_lstm(self, memory,time_step):
        if len(self.buffer)>64:
            for _ in range(32):
                self.update_q_cs()

        rewards = list(range(len(memory.rewards)))
        discounted_reward = 0
        mmm = 1-torch.tensor(memory.is_terminals).float().cuda(self.gpu)
                     
        rew = torch.tensor(memory.rewards).cuda(self.gpu).float()
        for i in reversed(range(len(memory.rewards))):
            discounted_reward = rew[i] + \
                self.gamma * mmm[i] * discounted_reward
        
            rewards[i] = discounted_reward.detach()

        rewards = torch.stack(rewards).reshape(16,-1)
        actions = torch.stack(memory.actions).cuda(self.gpu).float().detach().reshape(16,-1,self.action_dim)
        states = torch.stack(memory.states).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        nextstates=torch.stack(memory.nextstates).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        logprobs=torch.stack(memory.logprobs,0).detach().reshape(16,-1)
        with torch.no_grad():
            l=self.policy.lnet.get_l(torch.cat([states,actions,nextstates],2))
            value=self.policy.cnet_l(states,l.transpose(0,1),(self.hxzero,self.cxzero)).squeeze(-1)
            
        
        for _ in range(4):
            for index in BatchSampler(SubsetRandomSampler(range(16)), 4, False):
                new_logprobs,dist_entropy = self.evaluate(states[index].view(-1,self.input_dim),actions[index].view(-1,self.action_dim))
            
                #state_values=self.policy.critic(states[index]).squeeze(-1)
                state_values=self.policy.cnet_l(states[index],l[index].transpose(0,1),(self.hxzero[:,index,:].detach(),self.cxzero[:,index,:].detach())).squeeze(-1)


                ratios =torch.exp(new_logprobs-logprobs[index].detach().view(-1))
                advantages= rewards[index] -value[index].detach()
                surr1 = ratios * advantages.view(-1)
                surr2 = torch.clamp(ratios, 1-0.1, 1+0.1) * advantages.view(-1)
                #state_values=self.policy.critic(states[index].view(-1,self.input_dim))
                closs = self.MseLoss(state_values, rewards[index]).mean()
                aloss=-torch.min(surr1, surr2).mean()
                eloss=-0.01*dist_entropy.mean()
                loss=aloss+eloss
                self.opt_a.zero_grad()
                loss.backward()
                self.opt_a.step()
                self.opt_c_l.zero_grad()
                closs.backward()
                self.opt_c_l.step()



        f0 = open(self.logdir+'/closs_'+'.txt', 'a+')
        f1 = open(self.logdir+'/aloss_'+'.txt', 'a+')
        f2 = open(self.logdir+'/eloss_'+'.txt', 'a+')

        f0.writelines(str(closs.item())+'\n')
        f1.writelines(str(aloss.item())+'\n')
        f2.writelines(str(eloss.item())+'\n')

        f0.close()
        f1.close()
        f2.close()

    def update_lstm_gae(self, memory,time_step,lam=1):
        if len(self.buffer)>64:
            for _ in range(32):
                self.update_q_cs()
        actions = torch.stack(memory.actions).cuda(self.gpu).float().detach().reshape(16,-1,self.action_dim)
        states = torch.stack(memory.states).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        nextstates=torch.stack(memory.nextstates).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        logprobs=torch.stack(memory.logprobs,0).detach().reshape(16,-1)
        rewards = list(range(len(memory.rewards)))
        with torch.no_grad():
            l=self.policy.lnet.get_l(torch.cat([states,actions,nextstates],2))
            value=self.policy.cnet_l(states,l.transpose(0,1),(self.hxzero,self.cxzero)).squeeze(-1)
        prev=value.view(-1).cpu().numpy().tolist()
        prev.append(0)
        rew = torch.tensor(memory.rewards).cuda(self.gpu).float()
        mmm = 1-torch.tensor(memory.is_terminals).float().cuda(self.gpu)
        gae=0
        for i in reversed(range(len(memory.rewards))):
            delta= rew[i] + self.gamma * mmm[i] * prev[i+1]-prev[i]
            gae=delta+self.gamma*lam*mmm[i]*gae
            rewards[i] = gae + prev[i]
        rewards = torch.stack(rewards, 0).reshape(16,-1).detach()

       
        for _ in range(4):
            for index in BatchSampler(SubsetRandomSampler(range(16)), 4, False):
                new_logprobs,dist_entropy = self.evaluate(states[index].view(-1,self.input_dim),actions[index].view(-1,self.action_dim))
            
                #state_values=self.policy.critic(states[index]).squeeze(-1)
                state_values=self.policy.cnet_l(states[index],l[index].transpose(0,1),(self.hxzero[:,index,:].detach(),self.cxzero[:,index,:].detach())).squeeze(-1)


                ratios =torch.exp(new_logprobs-logprobs[index].detach().view(-1))
                advantages= rewards[index] -value[index].detach()
                surr1 = ratios * advantages.view(-1)
                surr2 = torch.clamp(ratios, 1-0.1, 1+0.1) * advantages.view(-1)
                #state_values=self.policy.critic(states[index].view(-1,self.input_dim))
                closs = self.MseLoss(state_values, rewards[index]).mean()
                aloss=-torch.min(surr1, surr2).mean()
                eloss=-0.01*dist_entropy.mean()
                loss=aloss+eloss
                self.opt_a.zero_grad()
                loss.backward()
                self.opt_a.step()
                self.opt_c_l.zero_grad()
                closs.backward()
                self.opt_c_l.step()

        f0 = open(self.logdir+'/closs_'+'.txt', 'a+')
        f1 = open(self.logdir+'/aloss_'+'.txt', 'a+')
        f2 = open(self.logdir+'/eloss_'+'.txt', 'a+')

        f0.writelines(str(closs.item())+'\n')
        f1.writelines(str(aloss.item())+'\n')
        f2.writelines(str(eloss.item())+'\n')

        f0.close()
        f1.close()
        f2.close()

    def update_single_gae(self, memory,time_step,lam=1):
        rewards = list(range(len(memory.rewards)))
        discounted_reward = 0
        mmm = 1-torch.tensor(memory.is_terminals).float()
       
                     

        
        actions = torch.stack(memory.actions).cuda(self.gpu).float().detach().reshape(16,-1,self.action_dim)
        states = torch.stack(memory.states,0).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        logprobs=torch.stack(memory.logprobs,0).detach().reshape(16,-1)
        with torch.no_grad():
            value=self.policy.critic(states.view(-1,self.input_dim))
        prev=value.squeeze(-1).cpu().numpy().tolist()
        prev.append(0)
        rew = torch.tensor(memory.rewards).cuda(self.gpu).float()
        gae=0
        for i in reversed(range(len(memory.rewards))):
            delta= rew[i] + self.gamma * mmm[i] * prev[i+1]-prev[i]
            gae=delta+self.gamma*lam*mmm[i]*gae
            rewards[i] = gae + prev[i]        
        rewards = torch.stack(rewards, 0).reshape(16,-1).detach()
        value=value.reshape(16,-1)

        for _ in range(4):
            for index in BatchSampler(SubsetRandomSampler(range(16)), 4, False):
                new_logprobs,dist_entropy = self.evaluate(states[index].view(-1,self.input_dim),actions[index].view(-1,self.action_dim))
                state_values=self.policy.critic(states[index].view(-1,self.input_dim))       

                ratios =torch.exp(new_logprobs-logprobs[index].detach().view(-1))
                advantages= rewards[index] -value[index].detach()
                surr1 = ratios * advantages.view(-1)
                surr2 = torch.clamp(ratios, 1-0.1, 1+0.1) * advantages.view(-1)
                #state_values=self.policy.critic(states[index].view(-1,self.input_dim))
                closs = self.MseLoss(state_values, rewards[index].view(-1)).mean()
                aloss=-torch.min(surr1, surr2).mean()
                eloss=-0.01*dist_entropy.mean()
                loss=aloss+eloss
                self.opt_a.zero_grad()
                loss.backward()
                self.opt_a.step()
                self.opt_c.zero_grad()
                closs.backward()
                self.opt_c.step()



        f0 = open(self.logdir+'/closs_'+'.txt', 'a+')
        f1 = open(self.logdir+'/aloss_'+'.txt', 'a+')
        f2 = open(self.logdir+'/eloss_'+'.txt', 'a+')

        f0.writelines(str(closs.item())+'\n')
        f1.writelines(str(aloss.item())+'\n')
        f2.writelines(str(eloss.item())+'\n')

        f0.close()
        f1.close()
        f2.close()


       
    def update_single(self, memory,time_step):
        rewards = list(range(len(memory.rewards)))
        discounted_reward = 0#self.value(memory.nextstate).squeeze()
        mmm = 1-torch.tensor(memory.is_terminals).float().cuda(self.gpu)
                     
        rew = torch.tensor(memory.rewards).cuda(self.gpu).float()
        for i in reversed(range(len(memory.rewards))):
            discounted_reward = rew[i] + \
                self.gamma * mmm[i] * discounted_reward
        
            rewards[i] = discounted_reward.detach()

        rewards = torch.stack(rewards, 0).reshape(16,-1)
        actions = torch.stack(memory.actions).cuda(self.gpu).float().detach().reshape(16,-1,self.action_dim)
        states = torch.stack(memory.states,0).cuda(self.gpu).float().detach().reshape(16,-1,self.input_dim)
        logprobs=torch.stack(memory.logprobs,0).detach().reshape(16,-1)
        value=self.policy.critic(states.view(-1,self.input_dim)).view(16,-1)
        
        for _ in range(4):
            for index in BatchSampler(SubsetRandomSampler(range(16)), 4, False):
                new_logprobs,dist_entropy = self.evaluate(states[index].view(-1,self.input_dim),actions[index].view(-1,self.action_dim))
                state_values=self.policy.critic(states[index].view(-1,self.input_dim))       

                ratios =torch.exp(new_logprobs-logprobs[index].detach().view(-1))
                advantages= rewards[index] -value[index].detach()
                surr1 = ratios * advantages.view(-1)
                surr2 = torch.clamp(ratios, 1-0.1, 1+0.1) * advantages.view(-1)
                #state_values=self.policy.critic(states[index].view(-1,self.input_dim))
                closs = self.MseLoss(state_values, rewards[index].view(-1)).mean()
                aloss=-torch.min(surr1, surr2).mean()
                eloss=-0.01*dist_entropy.mean()
                loss=aloss+eloss
                self.opt_a.zero_grad()
                loss.backward()
                self.opt_a.step()
                self.opt_c.zero_grad()
                closs.backward()
                self.opt_c.step()



        f0 = open(self.logdir+'/closs_'+'.txt', 'a+')
        f1 = open(self.logdir+'/aloss_'+'.txt', 'a+')
        f2 = open(self.logdir+'/eloss_'+'.txt', 'a+')

        f0.writelines(str(closs.item())+'\n')
        f1.writelines(str(aloss.item())+'\n')
        f2.writelines(str(eloss.item())+'\n')

        f0.close()
        f1.close()
        f2.close()


    def update_q_cs(self):
        if len(self.buffer)<self.batch_size:
            batch_size=len(self.buffer)
        else:
             batch_size=self.batch_size
        index=torch.tensor(np.random.choice(len(self.buffer),batch_size))

        #print(index)
        actions=torch.stack(self.buffer.actions)[index].cuda(self.gpu).float()
        states=torch.stack(self.buffer.states)[index].cuda(self.gpu).float()
        nextstates=torch.stack(self.buffer.nextstates)[index].cuda(self.gpu).float()
        rewards=torch.tensor(self.buffer.rewards)[index].cuda(self.gpu).float().unsqueeze(1)
         
        pre,l=self.policy.lnet(states,actions,nextstates)
        loss1=((nextstates-pre)**2).mean()
        #print(loss1)
        loglikeli=self.policy.qnet.loglikeli(torch.cat([states,actions],1),l.detach())
        #print(prob)
        loss2=-loglikeli
        self.opt_q.zero_grad()
        loss2.backward()

        self.opt_q.step()

        mi_est=self.policy.qnet.mi_est(torch.cat([states,actions],1),l)
        print(loss1,loss2,mi_est)
        loss_l=(loss1+mi_est)

        self.opt_l.zero_grad()
        loss_l.backward()
        nn.utils.clip_grad_norm(self.policy.lnet.parameters(), 5, norm_type=2)
        self.opt_l.step()


