#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(save_dir, joints_num):
    data_list = []
    joints_dirs = ["your_path/new_joint_vecs/"]

    for data_dir in joints_dirs:
        file_list = os.listdir(data_dir)
        for file in file_list:
            data = np.load(pjoin(data_dir, file), allow_pickle=True)
            if "humanml3d" in data_dir:
                if np.isnan(data).any():
                    print(file)
                    continue
                data_list.append(data)
            elif "motionfix_retarget" in data_dir:
                data = data.item()
                if np.isnan(data['source']).any() or np.isnan(data['target']).any():
                    print(file)
                    continue
                data_list.append(data['source'])
                data_list.append(data['target'])

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


# In[2]:


if __name__ == '__main__':
    save_dir = './'
    mean, std = mean_variance(save_dir, 22)

mean_path = './Mean.npy'
std_path = './Std.npy'

mean = np.load(mean_path)
std = np.load(std_path)

print(mean.shape, std.shape)


# In[1]:


import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance_motion(save_dir, joints_num):
    # sample_num = 0
    data_list = []
    joints_dirs = ["your_path/new_joints", "your_path/new_joints"]

    for data_dir in joints_dirs:
        file_list = os.listdir(data_dir)
        for file in file_list:
            data = np.load(pjoin(data_dir, file), allow_pickle=True)
            if "humanml3d" in data_dir or "HumanML3D" in data_dir:
                if np.isnan(data).any():
                    print(file)
                    continue
                # sample_num += 1
                if len(data.shape) == 2:
                    print(data.shape)
                    data_list.append(data[np.newaxis, ...])
                else:
                    data_list.append(data)
            elif "motionfix_retarget" in data_dir:
                data = data.item()
                if np.isnan(data['source']).any() or np.isnan(data['target']).any():
                    print(file)
                    continue

                if len(data['source'].shape) == 2:
                    print(data['source'].shape)
                    data_list.append(data['source'][np.newaxis, ...])
                else:
                    data_list.append(data['source'])

                if len(data['target'].shape) == 2:
                    print(data['target'].shape)
                    data_list.append(data['target'][np.newaxis, ...])
                else:
                    data_list.append(data['target'])

    data = np.concatenate(data_list, axis=0)
    # print(data.shape, sample_num, data.shape[0] / sample_num)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    np.save(pjoin(save_dir, 'mean_motion.npy'), Mean)
    np.save(pjoin(save_dir, 'std_motion.npy'), Std)

    return Mean, Std


# In[2]:


if __name__ == '__main__':
    save_dir = './'
    mean, std = mean_variance_motion(save_dir, 22)

mean_path = './mean_motion.npy'
std_path = './std_motion.npy'

mean = np.load(mean_path)
std = np.load(std_path)

print(mean.shape, std.shape)

