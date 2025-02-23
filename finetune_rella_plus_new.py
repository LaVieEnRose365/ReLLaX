import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import time
from transformers import set_seed
from dataset_new import generate_and_tokenize_prompt
from modeling_rella_plus_new import ReLLaModel
from dataset_new import get_text_and_ctr_dataset, LorecDataCollator
from ctr_base.config import Config
from functools import partial
from utils import lora_tsne, get_ctr_config

import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--output_path", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--model_name", type=str, default="vicuna-13b")
parser.add_argument("--ctr_model_path", type=str)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=256)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--test_range", type=str, default='all')
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--only_eval", action='store_true')
parser.add_argument("--dataset", type=str, default="BookCrossing")
parser.add_argument("--epochs", type=int, default=10)

#model args
parser.add_argument("--mode", type=str, default='origin')
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--layers", type=str, default="[q,v]")
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--enable_softmax", action="store_true")
parser.add_argument("--enable_sqrt_after_softmax", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ret", action="store_true")
parser.add_argument("--load_rella", action="store_true")
parser.add_argument("--load_rella_plus", action="store_true")
parser.add_argument("--ctr_dropout", type=float, default=0)
parser.add_argument("--ctr_out_layer", type=str, default='final')
parser.add_argument("--ctr_K", type=int, default=30)

# Here are args of prompt
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--train_type", type=str, default="simple")
parser.add_argument("--test_type", type=str, default="simple")

args = parser.parse_args()

assert args.train_type in ["simple", "sequential", "mixed", "high","all"]
assert args.test_type in ["simple", "sequential", "high"]
assert args.dataset in ["ml-1m", "BookCrossing", "GoodReads", "AZ-Toys", "ml-25m"]

data_path = f"Datasets/{args.dataset}/benchmark_proc_data/data"

t1 = time.time()
if args.layers == "[up,down,gate]":
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = 2
else:
    if args.K <= 15:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
    elif args.K <= 40:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
    else:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1


print('*'*70)
print(args)
print('*'*70)

transformers.set_seed(args.seed)

if args.train_type == "mixed":
    print(f"Shot: {args.train_size}")
    args.train_size *= 2
    print(f"Samples used: {args.train_size}")

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
# optimized for RTX 4090. for larger GPUs, increase some of these?

BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
EPOCHS = args.epochs  # we don't always need 3 tbh
LEARNING_RATE = args.lr  # the Karpathy constant
CUTOFF_LEN = 2048  # 256 accounts for about 96% of the data
LORA_R = args.lora_r
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = args.lora_dropout
VAL_SET_SIZE = args.val_size #2000
USE_8bit = True

if USE_8bit is True:
    warnings.warn("If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2")
        
DATA_PATH = {
    "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}_sampled.json"]), 
    # "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}.json"]), 
    # "val": '/'.join([args.data_path, f"valid/valid_{args.K}_{args.test_type}_sampled.json"]),
    "test": '/'.join([data_path, f"test/test_{args.K}_{args.test_type}.json"])
}
if args.train_type == "all" or args.dataset == 'ml-25m':
    DATA_PATH["train"] = '/'.join([data_path, f"train/train_{args.K}_{args.train_type}.json"])

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

print(world_size)
print(ddp)

if args.model_name == "vicuna-7b":
    args.model_path = "llm/models/vicuna-7b-v1.5"
    llama_config = {"model_name": "vicuna-7b", "model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 4096, 'intermediate_dim':11008, 'layer_num': 32}
    # args.per_device_train_batch_size *=2
    # args.per_device_eval_batch_size *=2
elif args.model_name == "vicuna-13b":
    args.model_path = "llm/models/vicuna"
    llama_config = {"model_name": "vicuna-13b", "model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 5120, 'intermediate_dim':13824, 'layer_num': 40}

print(args.per_device_train_batch_size)
print(args.per_device_eval_batch_size)
print(llama_config)

ctr_config = get_ctr_config(args)
ctr_config = Config.from_dict(ctr_config)
if args.ret:
    ctr_config.ctr_model_path = f"llm/models/ctr_model/{args.dataset}/sim/hist_len_{args.ctr_K}/model.pt"
else:
    ctr_config.ctr_model_path = f"llm/models/ctr_model/{args.dataset}/din/hist_len_{args.ctr_K}/model.pt"


# TARGET_MODULES = args.layers[1:-1].split(',')
TARGET_MODULES = ["q_proj", "v_proj"]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
if args.load_rella:
    shot_num = args.train_size
    if args.train_type == "mixed":
        shot_num /= 2

    print(f"{shot_num} shot")

    if args.dataset == "ml-1m":
        if shot_num == 8192:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_8192_mixed_sequential_K_30/checkpoint-64/pytorch_model.bin"
        elif shot_num == 4096:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_4096_mixed_sequential_K_30/checkpoint-32/pytorch_model.bin"
        elif shot_num == 2048:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_2048_mixed_sequential_K_30/checkpoint-16/pytorch_model.bin"
        elif shot_num == 1024:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_1024_mixed_sequential_K_30/checkpoint-8/pytorch_model.bin"
        elif shot_num == 512:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_512_mixed_sequential_K_30/checkpoint-4/pytorch_model.bin"
        elif shot_num == 256:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_256_mixed_sequential_K_30/checkpoint-2/pytorch_model.bin"
        elif shot_num == 65536:
            rella_lora_path = f"llm/lora-Vicuna/lr_0.001_shot_65536_sequential_sequential_K_30/checkpoint-256/pytorch_model.bin"
        else:
            raise NotImplementedError()
        
        if args.train_type == "simple":
            rella_lora_path = "llm/lora-Vicuna/lr_0.0003_shot_8192_simple_simple_K_30/checkpoint-32/pytorch_model.bin"

    elif args.dataset == "ml-25m":
        if shot_num == 8192:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_8192_mixed_high_K_30_10_bs128/checkpoint-128/pytorch_model.bin"
        elif shot_num == 4096:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_4096_mixed_high_K_30_10/checkpoint-32/pytorch_model.bin"
        elif shot_num == 2048:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_2048_mixed_high_K_30_10/checkpoint-16/pytorch_model.bin"
        elif shot_num == 1024:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_1024_mixed_high_K_30_10/checkpoint-8/pytorch_model.bin"
        elif shot_num == 512:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_512_mixed_high_K_30_10/checkpoint-4/pytorch_model.bin"
        elif shot_num == 256:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_256_mixed_high_K_30_10/checkpoint-2/pytorch_model.bin"
        elif shot_num == 65536:
            rella_lora_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_65536_sequential_sequential_K_30_10/checkpoint-256/pytorch_model.bin"
        else:
            raise NotImplementedError()
        
        if args.train_type == "simple":
            rella_lora_path = "llm/ml-25m_lora-Vicuna/lr_0.001_shot_65536_simple_simple_K_30_10/checkpoint-256/pytorch_model.bin"
        
    elif args.dataset == "BookCrossing":
        # rella_lora_path = f"llm/BookCrossing_lora-Vicuna/lr_0.0015_shot_15934_mixed_sequential_K_50_10_bs256/checkpoint-124/pytorch_model.bin"
        rella_lora_path = "llm/BookCrossing_lora-Vicuna/lr_0.001_shot_4096_mixed_sequential_K_60_10_bs128/checkpoint-128/pytorch_model.bin"
        if args.train_type == "simple":
            rella_lora_path = "llm/BookCrossing_lora-Vicuna/lr_0.001_shot_1024_simple_simple_K_60/checkpoint-32/pytorch_model.bin"
    else:
        raise NotImplementedError()
else:
    rella_lora_path = None
if args.load_rella_plus:
   	# if args.dataset == "ml-1m":
    if True:
    	rella_plus_path = os.path.join(args.output_path, "pytorch_model.bin")
    # elif args.dataset == "ml-25m":
    #     rella_plus_path = f"llm/ml-25m_lora-Vicuna/lr_0.001_shot_65536_sequential_sequential_K_30_10/checkpoint-256/pytorch_model.bin"
    # elif args.dataset == "BookCrossing":
    #     rella_plus_path = f"llm/BookCrossing_lora-Vicuna/lr_0.0015_shot_15934_mixed_sequential_K_50_10_bs256/checkpoint-124/pytorch_model.bin"
    # else:
    #     raise NotImplementedError()
else:
    rella_plus_path = None

if args.K > 20:
    use_gradient_checkpointing = True
else:
    use_gradient_checkpointing = False

tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, 
    add_eos_token=True, 
)

# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference
tokenizer.add_special_tokens({"additional_special_tokens": ["[hist]", "[target]"]})

model = ReLLaModel(
    llama_config, 
    lora_config, 
    ctr_config, 
    rella_lora_path=rella_lora_path,
    rella_plus_path=rella_plus_path,
    mode=args.mode, 
    USE_GC = use_gradient_checkpointing,
    tokenizer=tokenizer
)


CTR_DATA_PATH = f"Datasets/{args.dataset}/benchmark_proc_data"
if args.dataset == 'ml-25m':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "ml-25m", "sample_num": 90000, "hist_len": args.ctr_K, 'ret':args.ret, "prompt_K": args.K}
elif args.dataset == 'BookCrossing':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "BookCrossing", "sample_num": 10000, "hist_len": args.ctr_K, 'ret':args.ret, "prompt_K": args.K}
elif args.dataset == 'ml-1m':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "ml-1m", "sample_num": 70000, "hist_len" : args.ctr_K, 'ret':args.ret, "prompt_K": args.K}
elif args.dataset == 'GoodReads':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "GoodReads", "sample_num": 70000, "hist_len" : args.ctr_K, 'ret':args.ret, "prompt_K": args.K}


if args.mode in ["rella_plus", "ctr_prompt"]:
	insert_ctr_embedding = True
elif args.mode in ["origin", "lora_r"]:
	insert_ctr_embedding = False
else:
	raise NotImplementedError()
    
    
data = get_text_and_ctr_dataset(
    TEXT_DATA_PATH=DATA_PATH, 
    CTR_DATA_PATH=CTR_DATA_PATH, 
    train_size=args.train_size, 
    train_type=args.train_type,
    test_range=args.test_range,
    tokenizer=tokenizer, 
    CUTOFF_LEN=CUTOFF_LEN, 
    data_config=data_config,
	insert_ctr_embedding=insert_ctr_embedding
)
train_data = data['train']
test_data = data['test']
print("Data processed.")
print(data['train'])
print(data['test'])
print("Data loaded.")


now_max_steps = max((len(data["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)
MAX_STEPS = now_max_steps
print("MAX_STEPS: ", MAX_STEPS)


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    np.save(os.path.join(args.output_path, f"labels_{args.test_range}"), pre[1])
    np.save(os.path.join(args.output_path, f"preds_{args.test_range}"), pre[0])
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    np.save(os.path.join(args.output_path, f"results_{args.test_range}"), np.array([auc, ll, acc]))
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold
print(args.dataset)


# lora_tsne(model,llama_config)
set_seed(42)
model.LM.config.use_cache = False

class StopTrainingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # stop on the first epoch
        control.should_training_stop = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        # max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=False,
        logging_strategy="steps", 
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_safetensors=False,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_path,
        save_total_limit=30,
        # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        load_best_model_at_end=False,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
        weight_decay=args.weight_decay,
    ),
    data_collator=LorecDataCollator(tokenizer=tokenizer),
    # data_collator=transformers.DataCollatorForSeq2Seq(
    #     tokenizer, return_tensors="pt", padding='longest'
    # ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    # callbacks = [] if args.dataset == 'BookCrossing' else [StopTrainingCallback()],
)

if torch.__version__ >= "2" and sys.platform != "win32":
    print(1)
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

print("Start training...")

set_seed(42)

if args.test_range == "all":
    print(trainer.evaluate(eval_dataset=test_data))
else:
    trainer.train()

print("time1")
print(time.time()- t1)
print((time.time()-t1) / 3600)