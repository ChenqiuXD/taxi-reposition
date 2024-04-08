import os, sys

def get_output_folder(alg_name, mode='train', continue_num=0):
    """ Return save folder
     Assume folder in the parent_dir have suffix -run, e.g. null-run1
    Finds the highest run number and sets the output folder to that number + 1. 
    """
    parent_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(parent_dir, exist_ok=True)
    
    if mode=='train':
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('_run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, alg_name)
        parent_dir = parent_dir + '_run{}'.format(experiment_id)

    return parent_dir