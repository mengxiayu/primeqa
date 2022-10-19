# from data collator (index, target_ids, target_mask, passage_ids, passage_masks)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
import random
import json
import numpy as np
import transformers
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, ModelOutput
from typing import List, Optional, Tuple, Union, Dict, Any

class MoEBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.n_expert = None
 

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]: 

        # NOTE reshape inputs here
        if input_ids != None:
            if input_ids.dim() == 3:
                self.n_expert = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
        
        if attention_mask != None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if encoder_outputs != None: # when generate(). the model.get_encoder() was called in "_prepare_encoder_decoder_kwargs_for_generation"
        #     # reshape encoder outputs. 
        #     encoder_outputs = encoder_outputs.view(-1, encoder_outputs.size(-2), encoder_outputs.size(-1))

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # NOTE reshap decoder input ids here
        decoder_input_ids = decoder_input_ids.expand(self.n_expert, decoder_input_ids.size(0), decoder_input_ids.size(1)).transpose(1, 0).reshape(self.n_expert * decoder_input_ids.size(0), -1)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        """
        (B, N, S) -> (B*N, S) -> (B*N, S, hidden) -> (B*N, S, vocab_size) -> reshape to (B, N, S, vocab_size) -> combine N to 1, get (B, S)
        """
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias # (bsz * n_passages, seq_len, vocab_size)
        
        # reshape to 3 dimension and then sum
        lm_logits = lm_logits.view(-1, self.n_expert, lm_logits.size(-2), lm_logits.size(-1)) # (bsz, n_passages, seq_len, vocab_size)

        lm_logits = torch.sum(lm_logits, dim=1) # equally combine each expert
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def generate(self, input_ids, **gen_kwargs):
        self.n_expert = input_ids.size(1)
        return super().generate(
            input_ids,
            **gen_kwargs
        )
    def new_func(self):
        print("yes")
    

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        
        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        inputs_tensor = inputs_tensor.view(-1, inputs_tensor.size(-1)) # NOTE reshape input_ids 
        encoder_kwargs[model_input_name] = inputs_tensor
        encoder_kwargs["attention_mask"] = encoder_kwargs["attention_mask"].view(inputs_tensor.size())
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

        
    
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


