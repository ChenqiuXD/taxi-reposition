# Parameter settings
Here I record some successful parameters:

## ddpg
For double loop:  
>    input_args = ['--algorithm_name', algo, '--seed', '35', '--mode', 'train',     
                  '--is_two_loop',  
                  '--episode_length', '3', '--min_bonus', '0', '--max_bonus', '4', '--lr_drivers', '5e-2',   
                  '--lr', '1e-4', '--tau', '5e-3', '--buffer_size', '128', '--batch_size', '16',     
                  '--warmup_steps', '10', '--num_env_steps', '8000',    
                  '--min_epsilon', '0', '--max_epsilon', '0.2', '--decre_epsilon_episodes', '4000'] 

For single loop:  
>    input_args = ['--algorithm_name', algo, '--seed', '35', '--mode', 'train',   
                #   '--is_two_loop',  
                  '--episode_length', '10', '--min_bonus', '0', '--max_bonus', '4', '--lr_drivers', '5e-1',   
                  '--lr', '1e-4', '--tau', '5e-3', '--buffer_size', '128', '--batch_size', '16',   
                  '--warmup_steps', '3000', '--num_env_steps', '8000',  
                  '--min_epsilon', '0', '--max_epsilon', '0.2', '--decre_epsilon_episodes', '4000']
