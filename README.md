# ReLLaX: Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation

## Setup and Data preprocessing
The original ReLLa is [here](https://github.com/LaVieEnRose365/ReLLa), which shares the setup and data preprocessing process with [ReLLaX](https://arxiv.org/abs/2501.13344).

## Model Modifying
We propose CFLoRA, which allows fully interaction between LoRA atom components. 
~~~bash
cp modified_lora.py path_to_your_lora_file
cp modified_modeling_llama.py path_to_your_modeling_llama_file (e.g. /conda_envs/anaconda3/envs/ReLLaX/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py)
~~~

## Finetuning
~~~bash
python ft_script_rella_plus.py --lr 3e-4 --train_size 65536 --train_type sequential --test_type sequential --epochs 5 --bs 256 --mode rella_plus
~~~

## Inference
~~~bash
python ft_script_rella_plus.py --lr 3e-4 --train_size 65536 --train_type sequential --test_type sequential --epochs 5 --bs 256 --inference --ckpt 1 --mode rella_plus
~~~
