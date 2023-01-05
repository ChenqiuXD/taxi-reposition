# taxi-reposition
This is a repository used in the project "Incentive Design for driver-reposition problem using graph-based reinforcement learning". We explain the usage of each directory in the following diagram
* `environment`: The reinforcement learning environment which simulate the adpatation behavior of drivers. 
* `algorithms`: The algorithms used to design the incentive. Currently available: null, meta
* `result`: Visualization of the simulation result.   
* `tryout_single` is a simpler environment, which only examine one re-position, rather than focus on multi-step re-position problem as reinforcement learning.  

## Installation
- Install required packages (pytorch, numpy, matplotlib)  
- Open `run_this.py` file and check the variable `algo` and hyper-parameters in `input_args`  
- Run `run_this.py` and after completion, a result file would appear in the directory, named in format `algorithm_name_episodes_{num_env_steps}_length_{episode_length}_seed_{seed}_{"%m_%d_%H_%M"}` where the last term is the date time.  
- Drag or copy the result file to `results` folder. Change the `file_name` variable in the `plot.py` and run this script. After completion, you can see the graph generated in the this folder. 


## Result example: 
For the method `null`, we display two graphs: the first is the reward trajectory, the second is the drivers / demands ratio and bonuses design trajectory.  
> reward trajectory  
  ![Alt text](https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/null_episodes_10000_length_6_seed_35_01_03_15_43_cost_traj.png)  
> ratio and bonuses trajectory  
  ![Alt text](https://github.com/ChenqiuXD/taxi-reposition/blob/master/imgs/null_episodes_10000_length_6_seed_35_01_03_15_43_idle_drivers.png)

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