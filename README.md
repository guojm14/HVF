# HVF
Hindsight Value Function for Variance Reduction in Stochastic Dynamic Environment (IJCAI 2021)

require：
pytorch 1.4.0  
python 3.6.7  
mujoco_py 1.50.1.34  
mjpro150  
openmpi  4.0.3 

reduce_var/PPO.py: The algorithm.  
reduce_var/mlp.py: Model architectures. 

For plot: plotforv1.py gaeplot.py  
Main Fucntion: mpitraingae.py mpitraingae_th.py

How to get the experiment results?
1. mkdir result_setgae
2. mkdir modes_setgae 
3. mpiexec -n 27 python mpitraingae.py 
4. mpiexec -n 27 python mpitraingae_th.py

Then you can plot with results saved in  result_setgae:
Using gaeplot.py to compare lambda for GAE.
Using plotforv1.py to plot the learing curve figure.

For your own tasks, I recommend tuning parameters such as h_dim, l_dim. In addition, the estimation of mutual information is an important part of the algorithm. In this paper, we set the scale of both loss functions $L_{CLUB}$ and $L_{P}$ to 1. But in my experience, for some other tasks, it is necessary to adjust the scale of the loss functions (PPO.py line 390) and the structure of the variational distribution network(mlp.py line 86-100 e.g. remove or remain the tanh in p_logvar and modify the activation function).
