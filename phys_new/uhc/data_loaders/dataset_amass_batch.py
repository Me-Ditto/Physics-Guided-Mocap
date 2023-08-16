import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from PIL import Image
import os.path
import torch
import numpy as np
import torch.utils.data as data
import glob
import pickle as pk
import joblib
from collections import defaultdict
from tqdm import tqdm
import ipdb
from multiprocessing import Pool

from uhc.utils.math_utils import (
    de_heading,
    transform_vec,
    quaternion_multiply,
    quaternion_inverse,
    rotation_from_quaternion,
    ewma,
)
from uhc.utils.torch_ext import to_numpy

class DatasetAMASSBatch(data.Dataset):
    def __init__(self, cfg, data_mode="train", seed=0, multiproess=True):
        np.random.seed(seed)
        self.name = "DatasetAMASSBatch"
        self.cfg = cfg
        self.data = defaultdict(dict)
        self.data_raw = dict()
        self.base_rot = np.array([0.7071, 0.7071, 0.0, 0.0])
        self.rotrep = cfg.data_specs.get("rotrep", "euler")
        self.fr_num = cfg.data_specs.get("fr_num", 90)
        self.has_z = cfg.data_specs.get("has_z", True)
        self.dt = cfg.data_specs.get("dt", 1 / 30)
        self.multiproess = multiproess
        data_files = (
            cfg.data_specs.get("train_files_path")
            if data_mode == "train"
            else cfg.data_specs.get("test_files_path")
        )
        for f in data_files:
            processed_data, raw_data = self.preprocess_data(f)
            [self.data[k].update(v) for k, v in processed_data.items()]
            self.data_raw.update(raw_data)

        self.data_keys = list(self.data["pose_aa"].keys())

        self.traj_dim = self.data["pose_aa"][self.data_keys[0]].shape[1]
        self.freq_keys = []
        for k, traj in self.data["pose_aa"].items():
            self.freq_keys += [
                k for _ in range(np.ceil(traj.shape[0] / self.fr_num).astype(int))
            ]
        self.freq_keys = np.array(self.freq_keys)

        
    def preprocess_data(self, data_file):
        data_raw = joblib.load(data_file)
        data_processed = defaultdict(dict)
        all_data = list(data_raw.items())
        if self.multiproess:
            num_jobs = 20
            jobs = all_data
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
            job_args = [(jobs[i],) for i in range(len(jobs))]
            print(f"Reading data with {len(job_args)} threads")
            try:
                pool = Pool(num_jobs)  # multi-processing
                job_res = pool.starmap(self.process_data_list, job_args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            except Exception as e:
                import ipdb

                ipdb.set_trace()

            [[data_processed[k].update(v) for k, v in j.items()] for j in job_res]
        else:
            print(f"Reading data with 1 thread")
            data_processed = self.process_data_list(data_list=all_data)

        return data_processed, data_raw

    def process_data_list(self, data_list):
        data_processed = defaultdict(dict)
        # pbar = tqdm(all_data)
        for take, curr_data in data_list:
            pose_aa = curr_data["pose_aa"]
            seq_len = pose_aa.shape[0]
            if seq_len <= self.fr_num:
                continue

            data_processed["pose_aa"][take] = to_numpy(curr_data["pose_aa"])
            data_processed["pose_6d"][take] = to_numpy(curr_data["pose_6d"])
            data_processed["trans"][take] = to_numpy(curr_data["trans"])

        return data_processed

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))


    def __getitem__(self, index):
        # sample random sequence from data
        take_key = self.sample_keys[index]
        sample = self.get_sample_from_key(take_key, fr_start=-1)
        return sample

    def get_seq_len_by_key(self, key):

        return self.data["pose_aa"][key].shape[0]

    def get_seq_len_by_idx(self, idx):
        return self.data["pose_aa"][self.get_seq_key(idx)].shape[0]

    def get_seq_key(self, index):
        return self.data_keys[index]

    def get_sample_from_key(self, take_key, full_sample=False, fr_start=0):
        self.curr_key = take_key
        if full_sample:
            self.fr_start = fr_start = 0
            self.fr_end = fr_end = self.data["pose_aa"][take_key].shape[0]
        elif fr_start != -1:
            self.fr_start = fr_start
            self.fr_end = fr_end = fr_start + self.fr_num
        else:
            seq_len = self.get_seq_len_by_key(take_key)
            self.fr_start = fr_start = np.random.randint(0, seq_len - self.fr_num)
            self.fr_end = fr_end = fr_start + self.fr_num

        data_return = {}
        for k in self.data.keys():
            if k in ["of_files"]:
                continue
            data_return[k] = self.data[k][take_key][fr_start:fr_end]

        return data_return

    def sample_seq(
        self,
        num_samples=1,
        batch_size=1,
        use_freq=True,
        freq_dict=None,
        full_sample=False,
        sampling_temp=0.2,
        sampling_freq=0.75,
    ):
        start_idx = 0
        if use_freq:
            if freq_dict is None:
                self.chosen_key = chosen_key = np.random.choice(self.freq_keys)
            else:
                init_probs = np.exp(
                    -np.array(
                        [
                            ewma(np.array(freq_dict[k])[:, 0] == 1)
                            if len(freq_dict[k]) > 0
                            else 0
                            for k in freq_dict.keys()
                        ]
                    )
                    / sampling_temp
                )
                init_probs = init_probs / init_probs.sum()
                self.chosen_key = chosen_key = (
                    np.random.choice(self.data_keys, p=init_probs)
                    if np.random.binomial(1, sampling_freq)
                    else np.random.choice(self.data_keys)
                )
                seq_len = self.get_seq_len_by_key(chosen_key)

                ####################
                # perfs = np.array(freq_dict[chosen_key])
                # if len(perfs) > 0 and len(perfs[perfs[:, 0] != 1][:, 1]) > 0 and np.random.binomial(1, sampling_freq) and not full_sample:
                #     perfs = perfs[perfs[:, 0] != 1][:, 1]
                #     chosen_idx = np.random.choice(perfs)
                #     start_idx = np.random.randint(max(chosen_idx- 30, 0), min(chosen_idx + 30, seq_len - self.fr_num))
                #     # print(start_idx, chosen_key)
                # elif not full_sample:
                #     start_idx = np.random.randint(0, seq_len - self.fr_num)
                # else:
                #     start_idx = 0
                ####################
                if full_sample:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, seq_len - self.fr_num)

        else:
            self.chosen_key = chosen_key = np.random.choice(self.data_keys)

        self.curr_take_ind = self.data_keys.index(chosen_key)
        data_dict = self.get_sample_from_key(
            chosen_key, fr_start=start_idx, full_sample=full_sample
        )

        return {
            k: torch.from_numpy(v)[
                None,
            ]
            if not torch.is_tensor(v)
            else v[
                None,
            ]
            for k, v in data_dict.items()
        }

    def get_key_by_ind(self, ind):
        return self.data_keys[ind]

    def get_seq_by_ind(self, ind, full_sample=False):
        take_key = self.data_keys[ind]
        data_dict = self.get_sample_from_key(take_key, full_sample=full_sample)
        return {
            k: torch.from_numpy(v)[
                None,
            ]
            for k, v in data_dict.items()
        }

    def get_seq_by_key(self, take_key, full_sample=False):
        data_dict = self.get_sample_from_key(take_key, full_sample=full_sample)
        return {
            k: torch.from_numpy(v)[
                None,
            ]
            for k, v in data_dict.items()
        }

    def get_len(self):
        return len(self.data_keys)

    def sampling_loader(self, batch_size=8, num_samples=5000, num_workers=1, fr_num=80):
        self.fr_num = int(fr_num)
        self.sample_keys = np.random.choice(self.freq_keys, num_samples, replace=True)
        self.data_len = len(self.sample_keys)  # Change sequence length
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return loader

    def iter_loader(self, batch_size=8, num_workers=1, fr_num=80):
        # Not really iter...
        self.fr_num = int(fr_num)
        self.data_curr = [
            i for i in self.freq_keys if self.data["pose_aa"][i].shape[0] >= fr_num
        ]
        self.sample_keys = self.data_curr
        self.data_len = len(self.sample_keys)  # Change sequence length
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return loader

    def __len__(self):
        return self.data_len

    def iter_data(self):
        data = {}
        for take_key in self.data_keys:
            self.curr_key = take_key
            seq_len = self.data["pose_aa"][take_key].shape[
                0
            ]  # not using the fr_num at all
            data_return = {}

            for k in self.data.keys():
                data_return[k] = self.data[k][take_key]

            data[take_key] = {
                k: torch.from_numpy(v)[
                    None,
                ]
                for k, v in data_return.items()
            }
        return data

    def get_data(self):
        return self.data


if __name__ == "__main__":
    np.random.seed(0)
    from uhc.utils.config_utils.uhm_config import Config

    cfg = Config(cfg_id="uhm_init", create_dirs=False)

    dataset = DatasetAMASSBatch(cfg)
    for i in range(10):
        generator = dataset.sampling_loader(
            num_samples=5000, batch_size=1, num_workers=1
        )
        for data in generator:
            import pdb

            pdb.set_trace()
        print("-------")