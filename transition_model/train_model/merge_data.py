import os
from pathlib import Path
import numpy as np

run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "\\data")
print(run_dir)

# Merge into a list
data_list = []
for idx, file in enumerate(run_dir.iterdir()):
    if str(file.name).endswith('npy'):
        tmp = np.load(file, allow_pickle=True)
        for idx, data in enumerate(tmp):
            if data==0:
                print(file.name, " the empty idx is: ", idx, '\n\n\n\n\n')
                break
            # Uncomment this when using data_modified-traffic
            if np.any(data['traffic']<1000):
                print("Data has some traffic below 1000, probabily there is something wrong with data. ")
            data['traffic'] = data['traffic']/10000.0*500
            data_list.append(data)

# Concatenate the data into a single file. 
data_cat = np.array(data_list)
print(len(data_cat))

save_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "\\train_model\\data_cat.npy"
np.save(save_dir, data_cat)

print('done')