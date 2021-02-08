#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
import time
from tqdm import tqdm
import logging


class physonet2012_DataLoader(object):
    def __init__(self, args, fold_id=0, label='inhos_mortality', debug=False):
        self.root_path = os.path.abspath(os.path.dirname(__file__)) + "/"
        self.data_path = os.path.abspath(os.path.dirname(__file__)) + "/data/"


        if label not in ["inhos_mortality"]:
            raise AttributeError("No application for physionet2012: %s" % label)
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
            "stime": [],
            "tdata": [],
            "tmask": [],
            "labels": [],
        }
        if self.debug:
            indices = indices[:128]
        for index in tqdm(indices):
            if self.mode == "regular":
                data = self.get_regular_sample(index, label, length=self.args.max_timesteps,
                                               ffill=self.args.ffill, 
                                               ffill_steps = self.args.ffill_steps,
                                               standardization=self.args.standardization,
                                               # data_clip=self.args.data_clip,
                                               # data_clip_min=self.args.data_clip_min,
                                               # data_clip_max=self.args.data_clip_max,
                                               )
            else:
                raise NotImplementedError("[!]No such dataset mode: %s" % self.mode)
            for key in data.keys():
                result[key].append(data[key])
        for key in result.keys():
            result[key] = np.array(result[key])
        return result


    def _ffill_by_steps(self, input_x, input_mask, default=2):
        (tsize, fsize) = input_x.shape
        for f in range(fsize):
            last_ob = -1
            t_count = 0
            for t in range(tsize):
                if input_mask[t][f] != 0:
                    last_ob = input_x[t][f]
                    t_count = 0
                elif last_ob != -1 and input_mask[t][f] == 0 and t_count < default:
                    input_x[t, f] = last_ob
                    input_mask[t, f] = 1
                    t_count += 1

        return input_x, input_mask

    def transfer_info(self, info):
        for i in range(3):
            if info[i] == -1:
                info[i] = self.demo_norm['avg'][i]
            info[i] = float(info[i] - self.demo_norm['avg'][i])/self.demo_norm['std'][i]
        if info[3] + info[4] == -2:
            info[3] = 1
        for i in range(3,9):
            if info[i] == -1:
                info[i] = 0
        return info

    def transfer_label(self, data, label):
        if label == "inhos_mortality":
            return data[4]
        elif label == "length_of_stay_3":
            # return data[2]
            return int(data[2] <= 3)
        elif label == "length_of_stay_7":
            # return data[2]
            return int(data[2] <= 7)

    def get_regular_sample(self, index,label, length=48, ffill=False, ffill_steps=48, standardization=False, data_clip=False, data_clip_min=-1*float('inf'), data_clip_max=float('inf')):
        data = np.load(self.data_path + "%s.npz" % index)
        result = {}
        result['info'] = np.array(self.transfer_info(data['info']))

        regular_data = data['regular_data'][()]
        for key in ["stime","tdata","tmask"]:
            result[key] = regular_data[key][:length]

        if standardization:
            result['tdata'] = (result['tdata'] - self.time_norm['avg']) / self.time_norm['std']

        if data_clip:
            result['tdata'] = np.clip(result['tdata'],a_min=data_clip_min,a_max=data_clip_max)

        if ffill:
            result['tdata'], result['tmask'] = self._ffill_by_steps(result['tdata'], result['tmask'], ffill_steps)

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
