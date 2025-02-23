import re
import json
import h5py
import random
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers import DataCollator,DataCollatorForSeq2Seq,DataCollatorWithPadding
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from functools import partial
import random
import os
from itertools import chain


def insert_placeholder(prompt):
    modified_prompt = re.sub(r"(\(\d+(\.\d+)? stars?\))", r"\1 [hist]", prompt)
    modified_prompt = re.sub(r"\.\nNote", " [target].\nNote", modified_prompt)
    return modified_prompt


def generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN, insert_ctr_embedding=True):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "\
            f"USER: {data_point['input']} ASSISTANT: "
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )

    if insert_ctr_embedding:
        user_prompt = insert_placeholder(user_prompt)
        
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    ) - 1  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        # padding="max_length",
    )["input_ids"][:-1]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


def load_ctr_dataset(CTR_DATA_PATH, train_size, train_type, test_range, data_config):
    
    meta_data = json.load(open(os.path.join(CTR_DATA_PATH,'ctr-meta.json'), 'r'))
    field_names = meta_data["field_names"]
    feature_count = meta_data["feature_count"]
    feature_dict = meta_data["feature_dict"]
    feature_offset = meta_data["feature_offset"]
    # movie_id_to_title = meta_data["movie_id_to_title"]
    num_ratings = meta_data["num_ratings"]
    num_fields = len(field_names)
    num_features = sum(feature_count)

    offset = np.array(feature_offset).reshape(1, num_fields)

    # item_field_idx = field_names.index("ISBN")
    item_field_idx = data_config['item_field_idx']
    item_id_offset = feature_offset[item_field_idx]
    if data_config['dataset_name'] in ["ml-25m","ml-1m"]:
        rating_offset = num_features - 1
    elif data_config['dataset_name'] == "BookCrossing":
        rating_offset = num_features
    elif data_config['dataset_name'] == "GoodReads":
        rating_offset = num_features - 1
    else:
        raise NotImplementedError(f"Unsupported dataset: {data_config['dataset_name']}")

    num_features += num_ratings
    print(1)


    split_names = ["train", "test"]
    hist_ids = dict()
    hist_ratings = dict()
    hist_mask = dict()
    with h5py.File(os.path.join(CTR_DATA_PATH, f"ctr_{data_config['hist_len']}.h5"), "r") as f:
        X = {split: f[f"{split} data"][:] + offset for split in split_names}
        Y = {split: f[f"{split} label"][:] for split in split_names}

        # if not data_config['ret']:
        hist_ids = {split: f[f"{split} history ID"][:] + item_id_offset for split in split_names}
        if data_config['dataset_name'] == "ml-25m":
            hist_ratings = {split: f[f"{split} history rating"][:] * 2 + rating_offset for split in split_names}
        elif data_config['dataset_name'] in ["BookCrossing", "ml-1m", "GoodReads"]:
            hist_ratings = {split: f[f"{split} history rating"][:] + rating_offset for split in split_names}
        else:
            raise NotImplementedError(f"Unsupported dataset: {data_config['dataset_name']}")
        hist_mask = {split: f[f"{split} history mask"][:] for split in split_names}


        if data_config['dataset_name'] == 'ml-25m':
            print(hist_ids["train"].shape)
            print(hist_ids["test"].shape)
            # origin 60, not sampled
            sample_indices = {
                "train": np.load(os.path.join(CTR_DATA_PATH, '9w_train_idx.npy')),
                "test": np.load(os.path.join(CTR_DATA_PATH, 'test_idx.npy'))
            }

            for split in ["train", "test"]:
                X[split] = X[split][sample_indices[split]]
                Y[split] = Y[split][sample_indices[split]]
                hist_ids[split] = hist_ids[split][sample_indices[split]]
                hist_ratings[split] = hist_ratings[split][sample_indices[split]]
                hist_mask[split] = hist_mask[split][sample_indices[split]]
                
        print(2)


    # For prompt
    with h5py.File(os.path.join(CTR_DATA_PATH, f"ctr_{data_config['prompt_K']}.h5"), "r") as f:
        if train_type in ["mixed", "simple"]:
            llm_hist_ids_simple = {split: f[f"{split} history ID"][:] + item_id_offset for split in split_names}
            llm_hist_mask_simple = {split: f[f"{split} history mask"][:] for split in split_names}

            if data_config['dataset_name'] == 'ml-25m':
                print(llm_hist_ids_simple["train"].shape)
                print(llm_hist_ids_simple["test"].shape)

                # origin 30, not sampled
                for split in ["train", "test"]:
                    llm_hist_ids_simple[split] = llm_hist_ids_simple[split][sample_indices[split]]
                    llm_hist_mask_simple[split] = llm_hist_mask_simple[split][sample_indices[split]]

    print(22)



    if data_config['ret'] or train_type == "mixed":
        with h5py.File(os.path.join(CTR_DATA_PATH,f"ret_hist_{data_config['hist_len']}.h5"), "r") as f:
            hist_ids_ret = {split: f[f"{split} history ID ret"][:] + item_id_offset for split in split_names}
            if data_config['dataset_name'] == "ml-25m":
                hist_ratings_ret = {split: f[f"{split} history rating ret"][:] * 2 + rating_offset for split in split_names}
            elif data_config['dataset_name'] in ["BookCrossing", "ml-1m", "GoodReads"]:
                hist_ratings_ret = {split: f[f"{split} history rating ret"][:] + rating_offset for split in split_names}
            else:
                raise NotImplementedError(f"Unsupported dataset: {data_config['dataset_name']}")
            hist_mask_ret = {split: f[f"{split} history mask ret"][:] for split in split_names}  
    
        if data_config['dataset_name'] == "ml-25m":
            print(hist_ids_ret["train"].shape)
            print(hist_ids_ret["test"].shape)
            # ret 60, not sampled
            for split in ["train", "test"]:
                hist_ids_ret[split] = hist_ids_ret[split][sample_indices[split]]
                hist_ratings_ret[split] = hist_ratings_ret[split][sample_indices[split]]
                hist_mask_ret[split] = hist_mask_ret[split][sample_indices[split]]
    print(3)


    # For prompt
    with h5py.File(os.path.join(CTR_DATA_PATH,f"ret_hist_{data_config['prompt_K']}.h5"), "r") as f:
        llm_hist_ids_ret = {split: f[f"{split} history ID ret"][:] + item_id_offset for split in split_names}
        llm_hist_mask_ret = {split: f[f"{split} history mask ret"][:] for split in split_names}

        if data_config['dataset_name'] == "ml-25m":
            print(llm_hist_ids_ret["train"].shape)
            print(llm_hist_ids_ret["test"].shape)
            # ret 30, already sampled to 9w
            pass
    print(4)


    if train_type == "all" :
        new_X_train = X['train']
        new_Y_train = Y['train']
        new_hist_ids = hist_ids_ret['train']
        new_hist_ratings = hist_ratings_ret['train']
        new_hist_mask  = hist_mask_ret['train']
        # new_llm_hist_ids = llm_hist_ids_ret['train']
        # new_llm_hist_mask = llm_hist_mask_ret['train']

    else:
        if data_config['dataset_name'] == 'BookCrossing':  

            if train_type == 'mixed':
                random.seed(42)
                sample_indexs = random.sample(range(len(X['train'])), data_config['sample_num'])
                new_X_train = np.array(list(chain.from_iterable(zip(X['train'][sample_indexs], X['train'][sample_indexs]))))[:train_size]
                new_Y_train = np.array(list(chain.from_iterable(zip(Y['train'][sample_indexs], Y['train'][sample_indexs]))))[:train_size]
                new_hist_ids = np.array(list(chain.from_iterable(zip(hist_ids['train'][sample_indexs],hist_ids_ret['train'][sample_indexs]))))[:train_size]
                new_hist_ratings = np.array(list(chain.from_iterable(zip(hist_ratings['train'][sample_indexs], hist_ratings_ret['train'][sample_indexs]))))[:train_size]
                new_hist_mask = np.array(list(chain.from_iterable(zip(hist_mask['train'][sample_indexs], hist_mask_ret['train'][sample_indexs]))))[:train_size]
                new_llm_hist_ids = np.array(list(chain.from_iterable(zip(llm_hist_ids_simple['train'][sample_indexs],llm_hist_ids_ret['train'][sample_indexs]))))[:train_size]
                new_llm_hist_mask = np.array(list(chain.from_iterable(zip(llm_hist_mask_simple['train'][sample_indexs], llm_hist_mask_ret['train'][sample_indexs]))))[:train_size]

            if train_type == 'sequential':
                random.seed(42)
                sample_indexs_train = random.sample(range(len(X['train'])), data_config['sample_num'])
                new_X_train = X['train'][sample_indexs_train][:train_size]
                new_Y_train = Y['train'][sample_indexs_train][:train_size]
                new_hist_ids = hist_ids['train'][sample_indexs_train][:train_size]
                new_hist_ratings = hist_ratings['train'][sample_indexs_train][:train_size]
                new_hist_mask = hist_mask['train'][sample_indexs_train][:train_size]
                new_llm_hist_ids = llm_hist_ids_ret['train'][sample_indexs_train][:train_size]
                new_llm_hist_mask = llm_hist_mask_ret['train'][sample_indexs_train][:train_size]
                
            
            if train_type == "simple":
                random.seed(42)
                sample_indexs_train = random.sample(range(len(X['train'])), data_config['sample_num'])
                new_X_train = X['train'][sample_indexs_train][:train_size]
                new_Y_train = Y['train'][sample_indexs_train][:train_size]
                new_hist_ids = hist_ids['train'][sample_indexs_train][:train_size]
                new_hist_ratings = hist_ratings['train'][sample_indexs_train][:train_size]
                new_hist_mask = hist_mask['train'][sample_indexs_train][:train_size]
                new_llm_hist_ids = llm_hist_ids_simple['train'][sample_indexs_train][:train_size]
                new_llm_hist_mask = llm_hist_mask_simple['train'][sample_indexs_train][:train_size]


            X['test'] = X['test'][test_range[0]:test_range[1]]
            Y['test'] = Y['test'][test_range[0]:test_range[1]]

            hist_ids['test'] = hist_ids['test'][test_range[0]:test_range[1]]
            hist_ratings['test'] = hist_ratings['test'][test_range[0]:test_range[1]]
            hist_mask['test'] = hist_mask['test'][test_range[0]:test_range[1]]


            # if data_config['ret']:
            #     if train_type != "mixed":
            #         new_hist_ids = hist_ids_ret['train'][sample_indexs_train][:train_size]
            #         new_hist_ratings = hist_ratings_ret['train'][sample_indexs_train][:train_size]
            #         new_hist_mask = hist_mask_ret['train'][sample_indexs_train][:train_size]

            #     hist_ids['test'] = hist_ids_ret['test'][test_range[0]:test_range[1]]
            #     hist_ratings['test'] = hist_ratings_ret['test'][test_range[0]:test_range[1]]
            #     hist_mask['test'] = hist_mask_ret['test'][test_range[0]:test_range[1]]

            if train_type in ["mixed", "sequential"]:
                test_llm_hist_ids = llm_hist_ids_ret['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_ret['test'][test_range[0]:test_range[1]]
            elif train_type == "simple":
                test_llm_hist_ids = llm_hist_ids_simple['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_simple['test'][test_range[0]:test_range[1]]



        elif data_config['dataset_name'] == 'ml-25m':

            # only ret_30_hist is sampled to 9w
            if train_type == 'mixed':
                new_X_train = np.array(list(chain.from_iterable(zip(X['train'], X['train']))))[:train_size]
                new_Y_train = np.array(list(chain.from_iterable(zip(Y['train'], Y['train']))))[:train_size]
                new_hist_ids = np.array(list(chain.from_iterable(zip(hist_ids['train'],hist_ids_ret['train']))))[:train_size]
                new_hist_ratings = np.array(list(chain.from_iterable(zip(hist_ratings['train'], hist_ratings_ret['train']))))[:train_size]
                new_hist_mask = np.array(list(chain.from_iterable(zip(hist_mask['train'], hist_mask_ret['train']))))[:train_size]
                new_llm_hist_ids = np.array(list(chain.from_iterable(zip(llm_hist_ids_simple['train'], llm_hist_ids_ret['train']))))[:train_size]
                new_llm_hist_mask = np.array(list(chain.from_iterable(zip(llm_hist_mask_simple['train'], llm_hist_mask_ret['train']))))[:train_size]


            elif train_type == 'sequential' or train_type == 'high':
                new_X_train = X['train'][:train_size]
                new_Y_train = Y['train'][:train_size]

                new_hist_ids = hist_ids['train'][:train_size]
                new_hist_ratings = hist_ratings['train'][:train_size]
                new_hist_mask = hist_mask['train'][:train_size]
                new_llm_hist_ids = llm_hist_ids_ret['train'][:train_size]
                new_llm_hist_mask = llm_hist_mask_ret['train'][:train_size]

            elif train_type == "simple":
                new_X_train = X['train'][:train_size]
                new_Y_train = Y['train'][:train_size]

                new_hist_ids = hist_ids['train'][:train_size]
                new_hist_ratings = hist_ratings['train'][:train_size]
                new_hist_mask = hist_mask['train'][:train_size]
                new_llm_hist_ids = llm_hist_ids_simple['train'][:train_size]
                new_llm_hist_mask = llm_hist_mask_simple['train'][:train_size]

            print(5)   
            
            X['test'] = X['test'][test_range[0]:test_range[1]]
            Y['test'] = Y['test'][test_range[0]:test_range[1]]

            hist_ids['test'] = hist_ids['test'][test_range[0]:test_range[1]]
            hist_ratings['test'] = hist_ratings['test'][test_range[0]:test_range[1]]
            hist_mask['test'] = hist_mask['test'][test_range[0]:test_range[1]]

            if train_type == "simple":
                test_llm_hist_ids = llm_hist_ids_simple['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_simple['test'][test_range[0]:test_range[1]]
            else:
                test_llm_hist_ids = llm_hist_ids_ret['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_ret['test'][test_range[0]:test_range[1]]

            print(6)    

        elif data_config['dataset_name'] == 'ml-1m':  
            random.seed(42)
            sample_indexs_train = random.sample(range(len(X['train'])), data_config['sample_num'])
            sample_indexs_train = np.load(os.path.join(CTR_DATA_PATH, 'train_idx.npy'))
            
            if train_type == "mixed":
                new_X_train = np.array(list(chain.from_iterable(zip(X['train'][sample_indexs_train], X['train'][sample_indexs_train]))))[:train_size]
                new_Y_train = np.array(list(chain.from_iterable(zip(Y['train'][sample_indexs_train], Y['train'][sample_indexs_train]))))[:train_size]
                new_hist_ids = np.array(list(chain.from_iterable(zip(hist_ids['train'][sample_indexs_train],hist_ids_ret['train'][sample_indexs_train]))))[:train_size]
                new_hist_ratings = np.array(list(chain.from_iterable(zip(hist_ratings['train'][sample_indexs_train], hist_ratings_ret['train'][sample_indexs_train]))))[:train_size]
                new_hist_mask = np.array(list(chain.from_iterable(zip(hist_mask['train'][sample_indexs_train], hist_mask_ret['train'][sample_indexs_train]))))[:train_size]
                new_llm_hist_ids = np.array(list(chain.from_iterable(zip(llm_hist_ids_simple['train'][sample_indexs_train],llm_hist_ids_ret['train'][sample_indexs_train]))))[:train_size]
                new_llm_hist_mask = np.array(list(chain.from_iterable(zip(llm_hist_mask_simple['train'][sample_indexs_train], llm_hist_mask_ret['train'][sample_indexs_train]))))[:train_size]
                

            else:
                if data_config['ret']:
                    new_X_train = X['train'][sample_indexs_train][:train_size]
                    new_Y_train = Y['train'][sample_indexs_train][:train_size]
                    new_hist_ids = hist_ids_ret['train'][sample_indexs_train][:train_size]
                    new_hist_ratings = hist_ratings_ret['train'][sample_indexs_train][:train_size]
                    new_hist_mask = hist_mask_ret['train'][sample_indexs_train][:train_size]
                    hist_ids['test'] = hist_ids_ret['test'][test_range[0]:test_range[1]]
                    hist_ratings['test'] = hist_ratings_ret['test'][test_range[0]:test_range[1]]
                    hist_mask['test'] = hist_mask_ret['test'][test_range[0]:test_range[1]]

                    if train_type == "simple":
                        new_llm_hist_ids = llm_hist_ids_simple['train'][sample_indexs_train][:train_size]
                        new_llm_hist_mask = llm_hist_mask_simple['train'][sample_indexs_train][:train_size]
                    elif train_type == "sequential":
                        new_llm_hist_ids = llm_hist_ids_ret['train'][sample_indexs_train][:train_size]
                        new_llm_hist_mask = llm_hist_mask_ret['train'][sample_indexs_train][:train_size]
        
                else:
                    new_X_train = X['train'][sample_indexs_train][:train_size]
                    new_Y_train = Y['train'][sample_indexs_train][:train_size]
                    new_hist_ids = hist_ids['train'][sample_indexs_train][:train_size]
                    new_hist_ratings = hist_ratings['train'][sample_indexs_train][:train_size]
                    new_hist_mask = hist_mask['train'][sample_indexs_train][:train_size]
                    hist_ids['test'] = hist_ids['test'][test_range[0]:test_range[1]]
                    hist_ratings['test'] = hist_ratings['test'][test_range[0]:test_range[1]]
                    hist_mask['test'] = hist_mask['test'][test_range[0]:test_range[1]]

                    if train_type == "simple":
                        new_llm_hist_ids = llm_hist_ids_simple['train'][sample_indexs_train][:train_size]
                        new_llm_hist_mask = llm_hist_mask_simple['train'][sample_indexs_train][:train_size]
                    elif train_type == "sequential":
                        new_llm_hist_ids = llm_hist_ids_ret['train'][sample_indexs_train][:train_size]
                        new_llm_hist_mask = llm_hist_mask_ret['train'][sample_indexs_train][:train_size]

            if train_type in ["mixed", "sequential"]:
                test_llm_hist_ids = llm_hist_ids_ret['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_ret['test'][test_range[0]:test_range[1]]
            elif train_type == "simple":
                test_llm_hist_ids = llm_hist_ids_simple['test'][test_range[0]:test_range[1]]
                test_llm_hist_mask = llm_hist_mask_simple['test'][test_range[0]:test_range[1]]
            
            X['test'] = X['test'][test_range[0]:test_range[1]]
            Y['test'] = Y['test'][test_range[0]:test_range[1]]
            
       
        elif data_config['dataset_name'] == 'GoodReads':  
            random.seed(42)
            sample_indexs_train = random.sample(range(len(X['train'])), data_config['sample_num'])
            random.seed(42)
            sample_indexs_test = random.sample(range(len(X['test'])), test_range[1]-test_range[0])
            
            if data_config['ret']:
                new_X_train = X['train'][sample_indexs_train][:train_size]
                new_Y_train = Y['train'][sample_indexs_train][:train_size]
                new_hist_ids = hist_ids_ret['train'][sample_indexs_train][:train_size]
                new_hist_ratings = hist_ratings_ret['train'][sample_indexs_train][:train_size]
                new_hist_mask = hist_mask_ret['train'][sample_indexs_train][:train_size]
                new_llm_hist_ids = llm_hist_ids_ret['train'][sample_indexs_train][:train_size]
                new_llm_hist_mask = llm_hist_mask_ret['train'][sample_indexs_train][:train_size]

                hist_ids_ret['test'] = hist_ids_ret['test'][sample_indexs_test][test_range[0]:test_range[1]]
                hist_ratings_ret['test'] = hist_ratings_ret['test'][sample_indexs_test][test_range[0]:test_range[1]]
                hist_mask_ret['test'] = hist_mask_ret['test'][sample_indexs_test][test_range[0]:test_range[1]]
                llm_hist_ids_ret['test'] = llm_hist_ids_ret['test'][sample_indexs_test][test_range[0]:test_range[1]]
                llm_hist_mask_ret['test'] = llm_hist_mask_ret['test'][sample_indexs_test][test_range[0]:test_range[1]]
       
            else:
                print('ctr not ret')
                new_X_train = X['train'][sample_indexs_train][:train_size]
                new_Y_train = Y['train'][sample_indexs_train][:train_size]
                new_hist_ids = hist_ids['train'][sample_indexs_train][:train_size]
                new_hist_ratings = hist_ratings['train'][sample_indexs_train][:train_size]
                new_hist_mask = hist_mask['train'][sample_indexs_train][:train_size]
                new_llm_hist_ids = llm_hist_ids['train'][sample_indexs_train][:train_size]
                new_llm_hist_mask = llm_hist_mask['train'][sample_indexs_train][:train_size]

                hist_ids['test'] = hist_ids['test'][sample_indexs_test][test_range[0]:test_range[1]]
                hist_ratings_ret['test'] = hist_ratings['test'][sample_indexs_test][test_range[0]:test_range[1]]
                hist_mask['test'] = hist_mask['test'][sample_indexs_test][test_range[0]:test_range[1]]
                llm_hist_ids['test'] = llm_hist_ids['test'][sample_indexs_test][test_range[0]:test_range[1]]
                llm_hist_mask['test'] = llm_hist_mask['test'][sample_indexs_test][test_range[0]:test_range[1]]
            
            X['test'] = X['test'][sample_indexs_test][test_range[0]:test_range[1]]
            Y['test'] = Y['test'][sample_indexs_test][test_range[0]:test_range[1]]


    ctr_data = {
        'train': {
            "X": new_X_train,
            "Y": new_Y_train,
            "hist_ids": new_hist_ids,
            "hist_ratings": new_hist_ratings,
            "hist_mask": new_hist_mask,
            "llm_hist_ids": new_llm_hist_ids,
            "llm_hist_mask": new_llm_hist_mask,
        },
        'test': {
            "X": X['test'],
            "Y": Y['test'],
            "hist_ids": hist_ids['test'],
            "hist_ratings": hist_ratings['test'],
            "hist_mask": hist_mask['test'],
            "llm_hist_ids": test_llm_hist_ids,
            "llm_hist_mask": test_llm_hist_mask,
        }
    }

    print(7)

    return ctr_data


class LorecDataset(Dataset):
    """ Lorec Dataset
    The LorecDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the templating step.
    """
    def __init__(self, text_data, ctr_data):
        self.text_data = text_data
        self.ctr_data = ctr_data
    
    def __getitem__(self, idx):
        return self.text_data[idx], self.ctr_data["X"][idx], self.ctr_data["Y"][idx], self.ctr_data["hist_ids"][idx], self.ctr_data["hist_ratings"][idx], self.ctr_data["hist_mask"][idx], \
        self.ctr_data["llm_hist_ids"][idx], self.ctr_data["llm_hist_mask"][idx]
    
    def __len__(self):
        return len(self.text_data)
        # return 16
    

def get_text_and_ctr_dataset(TEXT_DATA_PATH, CTR_DATA_PATH, train_size, train_type,test_range, tokenizer, CUTOFF_LEN, data_config, insert_ctr_embedding=True):
    text_data = load_dataset("json", data_files=TEXT_DATA_PATH)
    if train_type != "all":
        text_data["train"] = text_data["train"].select(range(train_size))

    start,end = 0,len(text_data['test'])
    if test_range != "all":
        print(f"total test num: {len(text_data['test'])}")
        print(f"test range: {test_range}")
        start = int(test_range.strip().split(":")[0])
        end = int(test_range.strip().split(":")[1])
        if end == -1:
            end = len(text_data["test"])
        text_data["test"] = text_data["test"].select([i for i in range(start, end)])
    test_range = [start,end]

    text_test_data = text_data['test'].map(partial(generate_and_tokenize_prompt,tokenizer=tokenizer, CUTOFF_LEN=CUTOFF_LEN, insert_ctr_embedding=insert_ctr_embedding))
    # val_data = data["val"].map(generate_and_tokenize_prompt)
    text_train_data = text_data["train"].map(partial(generate_and_tokenize_prompt,tokenizer=tokenizer, CUTOFF_LEN=CUTOFF_LEN, insert_ctr_embedding=insert_ctr_embedding))
    used_columns = ['input_ids','labels','attention_mask']
    train_unused_columns = list(set(text_train_data.column_names) - set(used_columns))
    text_unused_columns = list(set(text_test_data.column_names) - set(used_columns))
    text_train_data = text_train_data.remove_columns(text_unused_columns)
    text_test_data = text_test_data.remove_columns(train_unused_columns)
    print("Text Data processed.")
    
    ctr_data = load_ctr_dataset(CTR_DATA_PATH, train_size, train_type, test_range, data_config)
    ctr_train_data = ctr_data['train']
    ctr_test_data = ctr_data['test']
    print("ctr loaded.")
    train_data = LorecDataset(text_train_data, ctr_train_data)
    test_data = LorecDataset(text_test_data, ctr_test_data)
    print("dataset ok.")

    data = {
        'train': train_data,
        'test': test_data
    }
    return data


class LorecDataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text_data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding='longest')

    def __call__(self, features: List) -> Dict[str,any]:
        text_data = [f[0] for f in features]
        batch = self.text_data_collator.__call__(text_data)
        ctr_batch = {
            "X": torch.tensor(np.array([f[1] for f in features])).long(),
            "Y": torch.tensor(np.array([f[2] for f in features])).long(),
            "hist_ids": torch.tensor(np.array([f[3] for f in features])).long(),
            "hist_ratings": torch.tensor(np.array([f[4] for f in features])).long(),
            "hist_mask": torch.tensor(np.array([f[5] for f in features])).long(),
            "llm_hist_ids": torch.tensor(np.array([f[6] for f in features])).long(),
            "llm_hist_mask": torch.tensor(np.array([f[7] for f in features])).long()
        }
        batch.update(ctr_batch)
        return batch
