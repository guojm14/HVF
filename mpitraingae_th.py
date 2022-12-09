#!/usr/bin/python3


import argparse
import copy
import math
import os
import pdb
import sys
from datetime import datetime as dt
from time import time
import random
import numpy as np
import torch
import torch.nn as nn
from reduce_var.PPO import PPO, Memory
import copy
import gym
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

rank = comm.Get_rank()
size = comm.Get_size()
#lamlist=[1,0.95,0.9,0.8,0.6,0.2]
lamlist=[1,0.99,0.98,0.96,0.92,0.84,0.68,0.36,0]
lam=lamlist[rank//3]
index=rank%3
rungpu=(rank+4)%8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=str, default="nothing", help="load_name")
    parser.add_argument("--gpu", type=int, default=4, help="num_gpu")
    parser.add_argument("--agent", type=int, default=4, help="num_agent")

    args = parser.parse_args()

    ######################## Hyperparameters ######################
    action_dim = 7
    max_timesteps = 4000000

    update_timestep = 100  # update policy every n timesteps
    # constant std for action distribution (Multivariate Normal)
    K_epochs = 4           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99        # discount factor

    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)

    #jobname = 'PPOreach_4_0.05_bias_gae0.2'
    jobname='PPOthrower_5_0.15_bias_gae'+str(rank//3)+'_'+str(rank%3)
    loadjobname = args.l
 
    log_folder = 'result_setgae/result_'+jobname
    model_folder = 'models_setgae/models_'+jobname

    env=gym.make('Thrower-v2')
    env = gym.wrappers.FlattenObservation(
    env)
    env.env._max_episode_steps=50
    try:
        os.mkdir(log_folder)
        os.mkdir(model_folder)
    except:
        print('mkdir fail!')
    ################################################################
    
    
    obs=env.reset()

    agent=PPO(action_dim, 23, lr, betas, gamma,
           gpu=rungpu,h_dim=256,log_folder=log_folder)


    memory = Memory()
    t = 0
    action_count = [0, 0, 0, 0, 0, 0]
    action_res = [[], [], [], [], [], []]
    res_t = []



    #print('scatter firt obs',rank,obs)
    state=torch.tensor(obs).float().view(-1)

    time_step = 0
    ep=0
    count=0
    for i in range(max_timesteps):
        count+=1
        #print(state)
        action=agent.select_action(state, memory)
   
        time_step += 1
        obs,reward,done,info=env.step(action.numpy().tolist())
        #print(reward)
        memory.nextstates.append(torch.tensor(obs).float().view(-1))
        if len(agent.buffer)<agent.buffer_size:
            agent.buffer.add(state,action,reward,torch.tensor(obs).float().view(-1))
        else:
            agent.buffer.pop()
            agent.buffer.add(state,action,reward,torch.tensor(obs).float().view(-1))
            
        t+=reward
        state=torch.tensor(obs).float().view(-1)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        if done:
            ep+=1
            #print(memory.nextstates)
            obs=env.reset()
            state=torch.tensor(obs).float().view(-1)
            if ep%16==0:
                agent.update_lstm_gae(memory,ep,lam)
                memory.clear_memory()
            res_t.append(t)
            print(count)
            count=0
            print(i, 'agent', t)
            t = 0           
            print('#############'+jobname +
                      '##########################################')
                #print(res_t)
            if ep % 1000 == 0:
                #print(res_t)
                f = open(log_folder+"/t"+".txt", "a+")
                for line in res_t:
                    f.writelines(str(line)+"\n")
                f.close()

                res_t = []
