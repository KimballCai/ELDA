#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import time
from tqdm import tqdm
import logging


class MIMIC3_DataLoader(object):
    def __init__(self, args, fold_id=0, label='inhos_mortality', debug=False):
        print("root_path: " + os.path.abspath(os.path.dirname(__file__)))
        self.root_path = os.path.abspath(os.path.dirname(__file__)) + "/"
        self.data_path = os.path.abspath(os.path.dirname(__file__)) + "/data/"


        if label not in ["inhos_mortality"]:
            raise AttributeError("No application for MIMIC3: %s" % label)
        else:
            logging.info("fold path: %s_folds.npz"%(label))
            folds_info = np.load(self.root_path + label+ "_folds.npz")


        self.fold = folds_info["fold_tvt"][fold_id]
        logging.info("train: %s" % str(self.fold[0][:5]))
        self.input_dim = int(folds_info['input_dim'])
        self.info_dim = int(folds_info['info_dim'])
        self.output_dim = 1

        self.args = args
        self.debug = debug
        self.max_timesteps = args.max_timesteps
        self.mode = args.dataset_mode

        if self.mode in ['regular']:
            time_norm_info = folds_info["regular_norm"][()]
        else:
            raise NotImplementedError("[!]No such dataset mode: %s"%self.mode)
        self.time_norm = {
            "max": np.array(time_norm_info['max'][fold_id]),
            "min": np.array(time_norm_info['min'][fold_id]),
            "avg": np.array(time_norm_info['avg'][fold_id]),
            "std": np.array(time_norm_info['std'][fold_id]),
        }
        demo_norm_info = folds_info["demo_norm"][()]
        self.demo_norm = {
            "max": np.array(demo_norm_info['max'][fold_id]),
            "min": np.array(demo_norm_info['min'][fold_id]),
            "avg": np.array(demo_norm_info['avg'][fold_id]),
            "std": np.array(demo_norm_info['std'][fold_id]),
        }

        self.sets_id = {
            "train": 0,
            "valid": 1,
            "test": 2
        }

        self.sets = {
            "train": self.get_dataset(indices=self.fold[0],label=label),
            "valid": self.get_dataset(indices=self.fold[1],label=label),
            "test": self.get_dataset(indices=self.fold[2],label=label),
        }

    def get_dataset(self, indices, label):
        result = {
            "info": [],
            "diag": [],
            "stime": [],
            "tdata": [],
            "tmask": [],
            "labels": [],
        }
        if self.debug:
            indices = indices[:128]
        for index in tqdm(indices):
            index = index[:-4]
            if self.mode == "regular":
                data = self.get_regular_sample(index, label, length=self.args.max_timesteps, ffill=self.args.ffill,
                                               standardization=self.args.standardization,
                                               )
            else:
                raise NotImplementedError("[!]No such dataset mode: %s" % self.mode)
            for key in data.keys():
                result[key].append(data[key])
        for key in result.keys():
            result[key] = np.array(result[key])
        return result

    def _ffill(self,input_x, input_mask):
        (tsize, fsize) = input_x.shape
        for t in range(tsize):
            for f in range(fsize):
                if t == 0:
                    break
                elif input_mask[t, f] != 1 and input_mask[t - 1, f] == 1:
                    input_x[t, f] = input_x[t - 1, f]
                    input_mask[t, f] = 1
        return input_x, input_mask

    def transfer_info(self, info):
        if info[9] == -1:
            info[9] = 60.0
        info[9] = (info[9] - self.demo_norm['avg'][0])/self.demo_norm['std'][0]
        if info[10] == -1:
            info[10] = 160.0
        info[10] = (info[10] - self.demo_norm['avg'][1]) / self.demo_norm['std'][1]
        if info[11] == -1:
            info[11] = 60.0
        info[11] = (info[11] - self.demo_norm['avg'][2]) / self.demo_norm['std'][2]

        for i in range(9):
            if info[i] == -1:
                info[i] = 0
        return info

    def transfer_label(self, data, label):
        # "MORTALITY_INUNIT","MORTALITY_INHOSPITAL","MORTALITY","LOS_UNIT_UNIT","LOS_UNIT_DISCHARGE"
        if label == "inhos_mortality":
            return data[1]



    def get_regular_sample(self, index,label, length=48, ffill=False, standardization=False, data_clip=False, data_clip_min=-1*float('inf'), data_clip_max=float('inf')):
        data = np.load(self.data_path + "%s.npz" % index)
        result = {}
        result['info'] = np.array(self.transfer_info(data['info']))

        irregular_data = data['regular_data'][()]
        for key in ["stime","tdata","tmask"]:
            result[key] = irregular_data[key][:length]

        if standardization:
            result['tdata'] = (result['tdata'] - self.time_norm['avg']) / self.time_norm['std']

        if data_clip:
            result['tdata'] = np.clip(result['tdata'],a_min=data_clip_min,a_max=data_clip_max)

        if ffill:
            result['tdata'], result['tmask'] = self._ffill(result['tdata'], result['tmask'])

        result['tdata'] = result['tdata'] * result['tmask']

        result['labels'] = self.transfer_label(data['labels'], label)

        return result

    def get_generator(self, sub_set, shuffle, batch_size, return_whole):
        fold_len = len(self.fold[self.sets_id[sub_set]])
        if self.debug:
            fold_len = 128
        fold = np.array(range(fold_len))
        dataset = self.sets[sub_set]
        def _generator():
            while True:
                if shuffle:
                    np.random.shuffle(fold)
                batch_from = 0
                while batch_from < fold_len:
                    batch_fold = fold[batch_from:batch_from + batch_size]
                    input_info = dataset['info'][batch_fold]
                    if self.mode == 'linear':
                        input_x = dataset['tdata'][batch_fold]
                        input_mask = dataset['tmask'][batch_fold]
                        inputs = [input_info,input_x,input_mask]
                    elif self.mode in ['regular','irregular']:
                        input_t = dataset['stime'][batch_fold]
                        input_x = dataset['tdata'][batch_fold]
                        input_mask = dataset['tmask'][batch_fold]
                        inputs = [input_info, input_t, input_x, input_mask]
                    else:
                        raise NotImplementedError("[!]No such dataset mode: %s" % self.mode)
                    input_y = dataset['labels'][batch_fold]
                    yield (inputs,input_y)
                    batch_from += batch_size

        def _inputs_generator():
            for inputs, _ in _generator():
                yield inputs

        if not return_whole:
            return _inputs_generator()
        else:
            return _generator()

    def sub_steps(self, sub_set, batch_size):
        if self.debug:
            return (128 - 1) // batch_size + 1
        else:
            return (len(self.fold[self.sets_id[sub_set]]) - 1) // batch_size + 1

    def sub_label(self, sub_set):
        return self.sets[sub_set]['labels']

    def get_subset_size(self, set):
        return len(self.fold[self.sets_id[set]])

