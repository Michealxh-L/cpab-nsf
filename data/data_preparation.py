

#%%
import numpy as np

from data.power import PowerDataset

dataset = PowerDataset(split='data')
data = dataset.data
shape = dataset.data.shape
#%%
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.1
# print(shape[0]*train_ratio)
train_set, val_set, test_set = np.split(data,[int(shape[0]*train_ratio),int(shape[0]*(train_ratio+val_ratio))])
#%%
print(dataset.dim)
import os
import utils



splits = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
for key in splits.keys():
    path = os.path.join(utils.get_data_root(), 'power', '{}.npy'.format(key))
    print(path)
    path = path.replace('\\','/')
    print(path)
    np.save(path,splits[key])
print('Done!')
#%%
data = np.load(path)
print(data.shape)