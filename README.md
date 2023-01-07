# taxi-reposition
This is a repository used in the project "Incentive Design for driver-reposition problem using graph-based reinforcement learning". We explain the usage of each directory in the following diagram
* `environment`: The reinforcement learning environment which simulate the adpatation behavior of drivers. 
* `algorithms`: The algorithms used to design the incentive. Currently available: null, meta
* `output`: Visualization of the simulation result. Note that the file would not be uploaded to github, it would appear after running algorithms.   
* `tryout_single` is a simpler environment, which only examine one re-position, rather than focus on multi-step re-position problem as reinforcement learning.  

## Installation
- Install required packages (pytorch, numpy, matplotlib)  
- Open `run_this.py` file and check the variable `algo` and hyper-parameters in `input_args` (such as 'num_env_steps': number of episodes; 'episode_length': time steps in an episode, etc.)  
- Run `run_this.py` and after completion, a `output` folder would appear in the directory, and results named in format `algorithm_name_episodes_{num_env_steps}_length_{episode_length}_seed_{seed}_{"%m_%d_%H_%M"}` where the last term is the date time. Reward trajectory and drivers' re-position actions are also displayed.   


## Result example: 
For the method `null`, we display two graphs: the first is the reward trajectory, the second is the drivers / demands ratio and bonuses design trajectory.  
> reward trajectory  
  <!-- ![Alt text](https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/cost_traj.png)   -->

  <img src="https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/cost_traj.png" aling=center width="50 />  

> ratio and bonuses trajectory  
  <!-- ![Alt text](https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/idle_drivers.png) -->
  <img src="https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/idle_drivers.png", align=center width=50 />

## Packages required
pytorch 1.10.x   
numpy  
matplotlib    
~~torch-scatter~~  
~~torch-sparse~~  
~~torch_geometric~~   
~~sumo-related (sumolib, traci, etc)~~

## Reference
- Installation of miniconda, pytorch: 
   [https://www.cnblogs.com/guesswhoiscoming/p/16983361.html](https://www.cnblogs.com/guesswhoiscoming/p/16983361.html)  
- Workflow (or how to cooperate) in github:  
 [https://code.tutsplus.com/tutorials/how-to-collaborate-on-github--net-34267](https://code.tutsplus.com/tutorials/how-to-collaborate-on-github--net-34267)