import os
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import set_seed
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from typing import List, Optional, Tuple, Union
import math
from functools import partial
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
import types

from ctr_base.models import BaseModel
from utils import weight_init

class ReLLaModel(nn.Module):
    """
    Consist of one Language Model and a ctr-based lora generator.

    Args:
        config: LlamaConfig, CtrConfig
    """
    def __init__(
        self, 
        llama_config, 
        lora_config, 
        ctr_config, 
        rella_lora_path=None,
        rella_plus_path=None,
        mode='origin',
        USE_GC = False,
        tokenizer=None
    ):
        super().__init__()
        self.llama_config = llama_config
        self.lora_config = lora_config
        self.ctr_config = ctr_config
        self.mode = mode

        self.LM = LlamaForCausalLM.from_pretrained(
            llama_config["model_path"],
            load_in_8bit=llama_config["load_in_8bit"],
            device_map=llama_config["device_map"],
        )
        self.tokenizer = tokenizer
        new_vocab_size = self.LM.config.vocab_size + 2  # Add [hist] and [target] tokens
        self.LM.resize_token_embeddings(new_vocab_size)
        
        
        if llama_config["load_in_8bit"] is True:
            self.LM = prepare_model_for_int8_training(self.LM, use_gradient_checkpointing=USE_GC)
        self.LM = get_peft_model(self.LM, lora_config)
        if rella_lora_path is not None:
            print("Load lora weights")
            adapters_weights = torch.load(rella_lora_path, map_location=self.LM.device)
            set_peft_model_state_dict(self.LM, adapters_weights)
            print("lora load results")

        
        if mode in ["rella_plus", "lora_r"]:
            # FUNC_TEMPLATE = "def forward_{module}_{layer}(self, x, **kwargs): gate = 2 * self.lora_gate(kwargs['user_embed']).unsqueeze(1); output = self.original_forward(x); return output * gate" # `2 *` is important for gate to ensure the expected gating to be 1 instead of 0.5.
            # Since the newly added module is named as `lora_xxx` （ie, lora_gate), it will be automatically saved when calling self.LM.save_pretrained()

            for module in lora_config.target_modules:
                for layer in range(llama_config["layer_num"]):
                    # exec(FUNC_TEMPLATE.format(module=module, layer=layer)) # define the forward function wrapper
                    proj_net = getattr(self.LM.model.model.layers[layer].self_attn, module)
                    proj_net.lora_gate = nn.Sequential(
                        nn.LayerNorm(ctr_config.hidden_size),
                        nn.Linear(ctr_config.hidden_size, 2 * lora_config.r * lora_config.r),
                        nn.GELU(),
                        nn.Linear(2 * lora_config.r * lora_config.r, lora_config.r * lora_config.r),
                    ).to(proj_net.lora_A["default"].weight.device)
                    nn.init.kaiming_uniform_(proj_net.lora_gate[1].weight, a=math.sqrt(5))
                    proj_net.lora_gate[1].bias.data.zero_()
                    nn.init.kaiming_uniform_(proj_net.lora_gate[3].weight, a=math.sqrt(5))
                    proj_net.lora_gate[3].bias.data.zero_()
                    # proj_net.original_forward = proj_net.forward
                    # proj_net.forward = types.MethodType(globals()[f"forward_{module}_{layer}"], proj_net)

        if mode in ["rella_plus", "ctr_prompt"]:
            # Insert ctr item embeddings
            self.ctr_input_embedding_prj = nn.Sequential(
                nn.Linear(ctr_config.embed_size, llama_config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(llama_config["hidden_dim"], llama_config["hidden_dim"])
            )
            weight_init(self.ctr_input_embedding_prj)
            
        if rella_plus_path:
            print(f"Load {mode} weights")
            state_dict = torch.load(rella_plus_path, map_location=self.LM.device)
            assert len(state_dict) > 0
            for name, param in self.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
            print(f"Finish loading {mode} weights")
        
        for name, param in self.LM.named_parameters():
            param.requires_grad = True if "lora_gate" in name else False # We fix the LoRA A & B matrices.
            
        
        self.CTR_model = BaseModel.from_config(ctr_config)
        state_dict = torch.load(ctr_config.ctr_model_path)
        self.CTR_model.load_state_dict(state_dict)
        for name, param in self.CTR_model.named_parameters():
            param.requires_grad = False


        # This redirects the state_dict of ReLLaModel to LoRA module
        self.origin_state_dict = self.state_dict
        def new_state_dict():
            state_dict = self.origin_state_dict()
            to_return = {k: state_dict[k] for k in state_dict if "lora_gate" in k or "ctr_input_embedding_prj" in k} # 注意之后还要保存别的，比如input embed projector
            # print([(k, v.shape) for k, v in to_return.items()])
            return to_return
        self.state_dict = new_state_dict

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        X: Optional[torch.LongTensor] = None, 
        Y: Optional[torch.LongTensor] = None, 
        hist_ids: Optional[torch.LongTensor] = None,
        hist_ratings: Optional[torch.LongTensor] = None, 
        hist_mask: Optional[torch.LongTensor] = None, 
        llm_hist_ids: Optional[torch.LongTensor] = None,
        llm_hist_mask: Optional[torch.LongTensor] = None,
    ):
        if self.mode in ['origin']:
            return self.LM(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            ctr_hidden_states = self.CTR_model(X, Y, hist_ids, hist_ratings, hist_mask).detach()

            if self.mode == "lora_r":
                return self.LM(
                    input_ids=input_ids,
                    # inputs_embeds=llama_input_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    instance_embed=ctr_hidden_states,
                )
            
            llama_input_embeds = self.LM.get_input_embeddings()(input_ids).detach()

            hist_token_id = self.tokenizer("[hist]", add_special_tokens=False, return_tensors="pt").input_ids.item()
            target_token_id = self.tokenizer("[target]", add_special_tokens=False, return_tensors="pt").input_ids.item()

            # if self.mode in ["rella_plus", "ctr_prompt"]:
            ctr_target_embeds = self.CTR_model.embed(X[:, self.ctr_config.item_field_idx]).detach() # (B, E)
            ctr_target_embeds = self.ctr_input_embedding_prj(ctr_target_embeds)
            ctr_hist_embeds = self.CTR_model.embed(llm_hist_ids).detach() # (B, L, E)
            ctr_hist_embeds = self.ctr_input_embedding_prj(ctr_hist_embeds)

            for i in range(input_ids.shape[0]):
                target_idx = (input_ids[i] == target_token_id).nonzero().view(-1)
                llama_input_embeds[i, target_idx] = ctr_target_embeds[i]
                
                llama_hist_lens = llm_hist_mask[i].sum().item()
                hist_idx = (input_ids[i] == hist_token_id).nonzero().view(-1)
                llama_input_embeds[i, hist_idx] = ctr_hist_embeds[i][:llama_hist_lens]
            
            if self.mode == "ctr_prompt":
                return self.LM(
                    inputs_embeds=llama_input_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            elif self.mode == "rella_plus":
                return self.LM(
                    inputs_embeds=llama_input_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    instance_embed=ctr_hidden_states,
                )
   