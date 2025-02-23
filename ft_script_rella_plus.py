import subprocess
import argparse
import os
import time

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, nargs='+')
parser.add_argument("--train_size", type=int, nargs='+')
parser.add_argument("--train_type", type=str, nargs='+')
parser.add_argument("--test_type", type=str, nargs='+')
parser.add_argument("--dataset", type=str)
parser.add_argument("--K", type=int, nargs='+')
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, nargs='+')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--mode", type=str, default='origin')
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--model_name", type=str, default='vicuna-13b')
parser.add_argument("--layers", type=str, default='[q_proj,v_proj]')
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ret", action="store_true")
parser.add_argument("--load_rella", action="store_true")
parser.add_argument("--load_rella_plus", action="store_true")
parser.add_argument("--inference", action="store_true")
parser.add_argument("--test_range", type=str, default='all')
parser.add_argument("--ctr_dropout", type=float, default=0.1)
parser.add_argument("--ctr_out_layer", type=str, default='final')
parser.add_argument("--ctr_K", type=int, default=30)
parser.add_argument("--ckpt", type=int, nargs='+')


args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.makedirs(f"{args.dataset}_logs", exist_ok=True)
os.makedirs(f"{args.dataset}_lora-Vicuna", exist_ok=True)
print(os.environ["CUDA_VISIBLE_DEVICES"])
CUDA_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
# CUDA_NUM = 1

print(CUDA_NUM)
for ckpt in args.ckpt:
    for bs in args.bs:
        for test_type in args.test_type:
            for train_type in args.train_type:
                for lr in args.lr:
                    for train_size in args.train_size:
                        for K in args.K:
                            fp = f"{args.dataset}_rella_plus/Dataset_{args.dataset}_CUDA_NUM_{CUDA_NUM}_test_range_{args.test_range}_layers_{args.layers}_lr_{lr}_shot_{train_size}_{train_type}_{test_type}_K_{K}_ctr_K_{args.ctr_K}_{args.epochs}_bs{bs}_wd_{args.weight_decay}_model_{args.model_name}_mode_{args.mode}_lora_r_{args.lora_r}_ret_{args.ret}_lora_drpt_{args.lora_dropout}_new"
                            
                            if args.inference:
                                steps_per_epoch = int(train_size / bs)
                                if train_type == "mixed":
                                    steps_per_epoch *= 2
                                # dirs = [x for x in os.listdir(f"{args.dataset}_rella_plus") if f"bs{bs}" in x and f"ret_{args.ret}" in x and f"lr_{lr}" in x]
                                # assert len(dirs) == 1, len(dirs)
                                # fp = dirs[0]
                                fp = f"{fp}/checkpoint-{ckpt * steps_per_epoch}"
                            
                                if os.path.exists(os.path.join(fp, "preds_all.npy")):
                                    print(f"already done: {bs} {lr} {args.ret}")
                                    continue
                                fp = fp.replace(f"test_range_{args.test_range}", f"test_range_0:100")
                                if not os.path.exists(os.path.join(fp, "pytorch_model.bin")):
                                    print(f"not finish training: {bs} {lr} {args.ret}")
                                    continue
                            
                            run_py = f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} python -u finetune_rella_plus_new.py " if CUDA_NUM == 1 else f"torchrun --nproc_per_node={CUDA_NUM} --master_port=12345 finetune_rella_plus_new.py "
                            command = \
                                run_py + \
                                f"--lr {lr} "\
                                f"--dataset {args.dataset} "\
                                f"--train_size {train_size} "\
                                f"--train_type {train_type} "\
                                f"--test_range {args.test_range if not args.inference else 'all'} "\
                                f"--test_type {test_type} "\
                                f"--K {K} "\
                                f"--epochs {args.epochs} "\
                                f"--total_batch_size {bs} "\
                                f"--output_path {fp} "\
                                f"--mode {args.mode} "\
                                f"--weight_decay {args.weight_decay} "\
                                f"--model_name {args.model_name} "\
                                f"--layers {args.layers} "\
                                f"--lora_dropout {args.lora_dropout} "\
                                f"--ctr_dropout {args.ctr_dropout} "\
                                f"--lora_r {args.lora_r} "\
                                f"--ctr_out_layer {args.ctr_out_layer} "\
                                f"--ctr_K {args.ctr_K} "\
                                f"{'--resume' if args.resume else ''} "\
                                f"{'--ret' if args.ret else ''} "\
                                f"{'--load_rella' if args.load_rella else ''} "\
                                f"{'--load_rella_plus' if args.load_rella_plus else ''} "\
                                # f">> {fp}/log.txt"
                            print(command)
                            subprocess.run(command, shell=True)
