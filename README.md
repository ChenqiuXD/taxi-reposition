# taxi-reposition
This is a repository used in the project "Incentive Design for reducing pollution in ride-haling services via multi-agent model-based reinforcement learning". We explain the usage of each directory in the following diagram
* `transition_model`: we generate a GNN model which predict the state transition matrix P(s'|s,a). 
* `rl_algo`: we provide a playground for the multi-agent reinforcement learning (MARL) algorithms using the well-known marl environment multi-particle environment (mpe). Currently finished algorithms include iql, vdn, qmix(#TODO). 

## Run command
Here we explain how to use each packages
### transition model
Please first clone the repo to your local PC, then make sure that SUMO is installed.  
Go to website (https://sumo.dlr.de/docs/Downloads.php) and download corresponding version (I downloaded .msi version package). Then install the package and add the SUMO's bin directory to system's PATH following command from website (https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home). 

### rl algo
pass

## Packages required
pytorch 1.8.0+cu101  
torch_geometric 2.0.3  
dgl 0.6.1  

## Reference
The Graph neural network implementation is largely copied from kipf's repo:  
<https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py>