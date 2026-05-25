#!/usr/bin/env python
# coding: utf-8

# ## Extract Poses from Amass Dataset

# In[36]:


import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm



from human_body_prior.tools.omni_tools import copy2cpu as c2c

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[37]:


from human_body_prior.body_model.body_model import BodyModel

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

neutral_bm_path = './body_models/smplh/neutral/model.npz'
neutral_dmpl_path = './body_models/dmpls/neutral/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
faces = c2c(male_bm.f)

female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=neutral_dmpl_path).to(comp_device)


# In[38]:


paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('./amass_data'):
    if root == "./amass_data":
        continue
#     print(root, dirs, files)
#     for folder in dirs:
#         folders.append(os.path.join(root, folder))
    folders.append(root)
    for name in files:
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, name))


# In[39]:


save_root = './pose_data'
save_folders = [folder.replace('./amass_data', './pose_data') for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path] for name in dataset_names]


# In[40]:


trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
ex_fps = 20
def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True).item()
    fps = 30

    if bdata['gender'] == 'male':
        bm = male_bm
    elif bdata['gender'] == 'female':
        bm = female_bm
    else:
        bm = neutral_bm
    down_sample = int(fps / ex_fps)
#     print(frame_number)
#     print(fps)

    source_bdata_poses = bdata['motion_source']['poses'][::down_sample,...]
    source_bdata_trans = bdata['motion_source']['trans'][::down_sample,...]
    source_body_parms = {
            'root_orient': torch.Tensor(source_bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(source_bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(source_bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(source_bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas].unsqueeze(0), repeats=len(source_bdata_trans), axis=0)).to(comp_device),
        }
    # print(">>>>")
    # print(source_body_parms['root_orient'].shape)
    # print(source_body_parms['pose_body'].shape)
    # print(source_body_parms['pose_hand'].shape)
    # print(source_body_parms['trans'].shape)
    # print(source_body_parms['betas'].shape)
    # print(">>>>")
    with torch.no_grad():
        source_body = bm(**source_body_parms)
    source_pose_seq_np = source_body.Jtr.detach().cpu().numpy()
    source_pose_seq_np_n = np.dot(source_pose_seq_np, trans_matrix)
    source_pose_seq_np_n[..., 0] *= -1
    
    target_bdata_poses = bdata['motion_target']['poses'][::down_sample,...]
    target_bdata_trans = bdata['motion_target']['trans'][::down_sample,...]
    target_body_parms = {
            'root_orient': torch.Tensor(target_bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(target_bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(target_bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(target_bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas].unsqueeze(0), repeats=len(target_bdata_trans), axis=0)).to(comp_device),
        }
    with torch.no_grad():
        target_body = bm(**target_body_parms)
    target_pose_seq_np = target_body.Jtr.detach().cpu().numpy()
    target_pose_seq_np_n = np.dot(target_pose_seq_np, trans_matrix)
    target_pose_seq_np_n[..., 0] *= -1

    pose_data = {
        'source': source_pose_seq_np_n,
        'target': target_pose_seq_np_n
    }
    
    np.save(save_path, pose_data)
    return fps


# In[41]:


group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0


# This will take a few hours for all datasets, here we take one dataset as an example
# 
# To accelerate the process, you could run multiple scripts like this at one time.

# In[42]:


import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./amass_data', './pose_data')
        if save_path[-3:] == 'txt':
            continue
        save_path = save_path[:-3] + 'npy'
        fps = amass_to_pose(path, save_path)
        
    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
    time.sleep(0.5)

