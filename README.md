# taxi-reposition
This is a repository used in the project "Incentive Design for reducing pollution in ride-haling services via multi-agent model-based reinforcement learning". We explain the usage of each directory in the following diagram
* `transition_model`: we generate a GNN model which predict the state transition P(s'|s,a). 
* `rl_algo`: we test the multi-agent reinforcement learning (MARL) algorithms using the well-known marl environment multi-particle environment (mpe). Currently finished algorithms include iql, vdn. Qmix is still working. 
* `run_this`: contains the environment 'make_sumo_env.py' and agents.

## Installation
- Install SUMO (ref: https://sumo.dlr.de/docs/Downloads.php)
- Download training required packages: `pytorch-1.10.x`, `torch-scatter`, `torch-sparse`, `torch-geometric` (ref: https://www.cnblogs.com/guesswhoiscoming/p/15865709.html) and other required packages. (Strongly recommend run the code and install necessary packages when prompted.)
- Run `./transition_model/main.py` to test the availibility of sumo.  
- If you need run model-based algorithm, use codes in train_model firstly. Please refer to the following sections. 
- To run algorithms, please first open `./run_this/run_this.py`, change parameters in  the last __main__ function (such as "algorithm_name", "num_env_steps" etc) and then open a terminal and run command  
`python ./run_this/run_this.py`  
In windows (linux is still untested). If the terminal starts prompting and sumo started simulating, then the program runs succussfully. Wait until it ends, then you can obtain a file named 'xxx_episodes_{num_env_steps}\_length\_{episode_length}' as the recorded result. 
- Use `./plot_result.py` to plot results related to the observed result. Change the file name in the python code and then run it. After prompting 'done', you can obtain the png files which is information related to the results. 



## Run command
Here we explain how to use each packages
### transition model

### rl algo
pass

## Packages required
pytorch 1.10.x  
torch-scatter  
torch-sparse  
torch_geometric   
sumo-related (sumolib, traci, etc)

## Reference
