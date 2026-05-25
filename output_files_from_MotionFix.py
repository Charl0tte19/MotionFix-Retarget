#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import joblib
import numpy as np
import torch
from pathlib import Path

def cast_dict_to_tensors(d, device="cpu"):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v, device) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float().to(device)
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    else:
        return d
    
# print(dataset_dict_raw['000283'].keys()) # dict_keys(['motion_source', 'motion_target', 'text'])
# print(dataset_dict_raw['000283']['text'])
# print(dataset_dict_raw['000283']['motion_source'].keys()) # dict_keys(['rots', 'trans', 'joint_positions', 'timestamp'])

# print(dataset_dict_raw['000283']['motion_source']['timestamp'])
# print(dataset_dict_raw['000283']['motion_source']['trans'].shape)
# print(dataset_dict_raw['000283']['motion_source']['rots'].shape)
# print(dataset_dict_raw['000283']['motion_source']['joint_positions'].shape)


# In[35]:


not_train_keys = []
texts_output_dir = "./amass_data/"
if not os.path.exists(texts_output_dir):
    os.makedirs(texts_output_dir)
    
splits = ["val", "test", "train"]
for split in splits:
    datapath = "your_path/motionfix_{}.pth.tar".format(split)
    ds_db_path = Path(datapath)
    dataset_dict_raw = joblib.load(ds_db_path)
    data_dict = cast_dict_to_tensors(dataset_dict_raw)

    for key in data_dict.keys():
        name = int(key) + 40000
        name = str(name).zfill(6)

        if split == "val" or split == "test":
            not_train_keys.append(key)
            with open(os.path.join(texts_output_dir, split + ".txt"), "a") as f:
                f.write(name+"\n")
        else:
            if key not in not_train_keys:
                with open(os.path.join(texts_output_dir, split + ".txt"), "a") as f:
                    f.write(name+"\n")




# In[36]:


pose_hand = torch.zeros(1, 90)
motions_output_dir = "./amass_data/motion/"
texts_output_dir = "./amass_data/texts/"
if not os.path.exists(motions_output_dir):
    os.makedirs(motions_output_dir)
if not os.path.exists(texts_output_dir):
    os.makedirs(texts_output_dir)

splits = ["val", "test", "train"]
for split in splits:
    datapath = "your_path/motionfix_{}.pth.tar".format(split)
    ds_db_path = Path(datapath)
    dataset_dict_raw = joblib.load(ds_db_path)
    data_dict = cast_dict_to_tensors(dataset_dict_raw)

    for key in data_dict.keys():
        name = int(key) + 40000
        name = str(name).zfill(6)

        output = {}
        output["gender"] = "neutral"
        output["betas"] = torch.zeros(16)
        output["motion_source"] = {}
        output["motion_target"] = {}

        output["motion_source"]["trans"] = data_dict[key]["motion_source"]["trans"]
        poses = data_dict[key]["motion_source"]["rots"]
        

        poses = torch.cat((poses, pose_hand.repeat(poses.shape[0], 1)), dim=1)
        output["motion_source"]["poses"] = poses

        output["motion_target"]["trans"] = data_dict[key]["motion_target"]["trans"]
        poses = data_dict[key]["motion_target"]["rots"]
        poses = torch.cat((poses, pose_hand.repeat(poses.shape[0], 1)), dim=1)
        output["motion_target"]["poses"] = poses

        np.save(os.path.join(motions_output_dir, name + ".npy"), output)

        text = data_dict[key]["text"]
        with open(os.path.join(texts_output_dir, name + ".txt"), "w") as f:
            f.write(text)

