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
        os.makedirs(parent_dir, exist_ok=True)
    elif mode=='continue':
        name = ""
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            if str(folder_name).startswith(alg_name) and int(folder_name.split('_run')[-1])==continue_num:
                name = folder_name
                break
        
        if name == "":
            raise RuntimeError("No folder is output of algorithm {} with index {}. Please recheck the algorihtm and continue_num parameter in the run_this.py".format(alg_name, continue_num))
        else:
            parent_dir = os.path.join(parent_dir, name)
    
    return parent_dir