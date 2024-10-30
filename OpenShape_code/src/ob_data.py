import random

import torch
import numpy as np
import torch.utils.data as data
import copy

import yaml
from easydict import EasyDict

class Objaverse_lvis_openshape(data.Dataset):
    def __init__(self, config):

        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.picked_rotation_degrees = list(range(10))
        self.openshape_setting = config.openshape_setting
        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]


        self.data_list_file = config.PC_PATH
        self.pc_root = config.PC_PATH_ROOT

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')    # use both train and test data for pretraining

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Objaverse')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Objaverse')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            self.file_list.append({
                'cate_id': line.split(',')[0],
                'cate_name': line.split(',')[1],
                'model_id': line.split(',')[2],
                'point_path': self.pc_root + line.split(',')[3].replace('\n', '')
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='Objaverse')

        # exit()

        self.permutation = np.arange(self.npoints)

        self.uniform = False
        self.augment = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

        # self.template = "a point cloud model of {}."

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        if num < pc.shape[0]:
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        else:
            ran_sel = np.random.randint(0, pc.shape[0], num)
            pc = pc[ran_sel]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        cate_id, cate_name, model_id, point_path = sample['cate_id'], sample['cate_name'], sample['model_id'], sample['point_path']

        while True:
            try:
                openshape_data = np.load(point_path, allow_pickle=True).item()
                data = openshape_data['xyz'].astype(np.float32)
                rgb = openshape_data['rgb'].astype(np.float32)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying…")
                import time
                time.sleep(1)
            else:
                break

        if self.openshape_setting:
            data[:, [1, 2]] = data[:, [2, 1]]
            logging.info('flip yz')
            data = normalize_pc(data)
        else:
            data = self.pc_norm(data)
        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        cate_id = np.array([cate_id]).astype(np.int32)
        # print(data.shape, cate_id, cate_name)
        return data, cate_id, cate_name, rgb

    def __len__(self):
        return len(self.file_list)  
    