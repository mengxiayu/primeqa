# from data collator (index, target_ids, target_mask, passage_ids, passage_masks)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
import random
import json
import numpy as np
import transformers
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList
 
@dataclass
class Seq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    expert_weight: Optional[Tuple[torch.FloatTensor]] = None
    uncombined_logits: Optional[Tuple[torch.FloatTensor]] = None

class MoEBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.n_expert = None
        self.knowledge_selection = KnowledgeSelection(config.d_model)
    def set_moe_mode(self, mode, hard_weight):
        self.moe_mode = mode
        self.hard_weight = hard_weight

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
        is_debug: Optional[bool] = None,
        encoder_similarity: Optional[torch.FloatTensor] = None,
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
        # NOTE reshape decoder input ids here
        decoder_input_ids = decoder_input_ids.expand(self.n_expert, decoder_input_ids.size(0), decoder_input_ids.size(1)).transpose(1, 0).reshape(self.n_expert * decoder_input_ids.size(0), -1)
        # print("labels", labels.shape)
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
        uncombined_logits = self.lm_head(outputs[0]) + self.final_logits_bias # (bsz * n_passages, seq_len, vocab_size)
        
        # Option 1: knowledge selection
        if not return_dict:
            decoder_hidden, encoder_hidden = outputs
        else:
            decoder_hidden = outputs.last_hidden_state
            encoder_hidden = outputs.encoder_last_hidden_state
        # decoder_hidden = decoder_hidden.view(-1, self.n_expert, decoder_hidden.size(-2), decoder_hidden.size(-1))
        # encoder_hidden = encoder_hidden.view(-1, self.n_expert, encoder_hidden.size(-2), encoder_hidden.size(-1))

        # reweight experts
        if self.moe_mode == "similarity":
            if input_ids is not None:
                sequence_similarity = self.calculate_encoder_similarity(input_ids, encoder_hidden)
            else:
                assert encoder_similarity is not None
                sequence_similarity = encoder_similarity
            # print("similarity", sequence_similarity)
            lm_logits, expert_weight = self.knowledge_selection(uncombined_logits, encoder_hidden, decoder_hidden, self.n_expert, sequence_similarity, self.hard_weight)
        
        elif self.moe_mode == "equal":
            lm_logits = uncombined_logits

        # ensemble experts
        lm_logits = lm_logits.view(-1, self.n_expert, lm_logits.size(-2), lm_logits.size(-1)) # (bsz, n_passages, seq_len, vocab_size)
        lm_logits = torch.sum(lm_logits, dim=1)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if is_debug:
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
                expert_weight=expert_weight, # for debug purpose
                uncombined_logits=uncombined_logits # for debug purpose
            )
        else:
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

    def calculate_encoder_similarity(self, input_ids, encoder_hidden):
        # pooling: get embeddings of eos tokens
        eos_mask = input_ids.eq(self.config.eos_token_id).to(input_ids.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sequence_embedding = encoder_hidden[eos_mask, :].view(encoder_hidden.size(0), -1, encoder_hidden.size(-1))[
            :, -1, :
        ]
        # reshape to (B, N_expert, Hidden)
        sequence_embedding = sequence_embedding.reshape(-1, self.n_expert, encoder_hidden.size(-1))

        # calculate cosine similarity between each two inputs
        sequence_similarity = F.cosine_similarity(sequence_embedding[:, :, None], sequence_embedding[:, None, :], dim=-1) # (B, N, N)

        # only take the first row, which is the similarity between (input_i, Q)
        sequence_similarity = sequence_similarity[:, 0, :] 

        # reshape to (B x N_expert, 1)
        sequence_similarity = sequence_similarity.reshape(-1, 1) 
        return sequence_similarity  

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
        encoder_outputs = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
       

        # calculate similarity
        encoder_hidden = encoder_outputs.last_hidden_state
        if self.moe_mode == "similarity":
            sequence_similarity = self.calculate_encoder_similarity(inputs_tensor, encoder_hidden)
            model_kwargs["encoder_similarity"] = sequence_similarity # NOTE record input similarity
        else:
            model_kwargs["encoder_similarity"] = None

        return model_kwargs
    
    # called in generation()
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "encoder_similarity": kwargs["encoder_similarity"]
        }

    # for debug purpose
        # def greedy_search(
        #     self,
        #     input_ids: torch.LongTensor,
        #     logits_processor: Optional[LogitsProcessorList] = None,
        #     stopping_criteria: Optional[StoppingCriteriaList] = None,
        #     max_length: Optional[int] = None,
        #     pad_token_id: Optional[int] = None,
        #     eos_token_id: Optional[int] = None,
        #     output_attentions: Optional[bool] = None,
        #     output_hidden_states: Optional[bool] = None,
        #     output_scores: Optional[bool] = None,
        #     return_dict_in_generate: Optional[bool] = None,
        #     synced_gpus: Optional[bool] = False,
        #     **model_kwargs,
        # ) : # Not define output
        #     # init values
        #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        #     if max_length is not None:
        #         warnings.warn(
        #             "`max_length` is deprecated in this function, use"
        #             " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
        #             UserWarning,
        #         )
        #         stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        #     pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        #     eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        #     output_scores = output_scores if output_scores is not None else self.config.output_scores
        #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #     output_hidden_states = (
        #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #     )
        #     return_dict_in_generate = (
        #         return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        #     )

        #     # init attention / hidden states / scores tuples
        #     scores = () if (return_dict_in_generate and output_scores) else None
        #     decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        #     cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        #     decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        #     # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        #     if return_dict_in_generate and self.config.is_encoder_decoder:
        #         encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #         encoder_hidden_states = (
        #             model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #         )

        #     # keep track of which sequences are already finished
        #     unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        #     cur_len = input_ids.shape[-1]
        #     uncombined_sequences = []
        #     expert_selections = []
        #     # print("expert_selection", expert_selections.shape)
        #     this_peer_finished = False  # used by synced_gpus only
        #     while True:

        #         if synced_gpus:
        #             # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
        #             # The following logic allows an early break if all peers finished generating their sequence
        #             this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
        #             # send 0.0 if we finished, 1.0 otherwise
        #             dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
        #             # did all peers finish? the reduced sum will be 0.0 then
        #             if this_peer_finished_flag.item() == 0.0:
        #                 break

        #         # prepare model inputs
        #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        #         # forward pass to get next token
        #         outputs = self(
        #             **model_inputs,
        #             return_dict=True,
        #             output_attentions=output_attentions,
        #             output_hidden_states=output_hidden_states,
        #         )
        #         if synced_gpus and this_peer_finished:
        #             cur_len = cur_len + 1
        #             continue  # don't waste resources running the code we don't need

        #         next_token_logits = outputs.logits[:, -1, :]
        #         next_token_expert_weights = outputs.expert_weight
        #         next_token_uncombined_logits = outputs.uncombined_logits[:, -1, :]
                
        #         # print("logits", next_token_logits.shape)
        #         # print("uncombined logits", next_token_uncombined_logits.shape)

        #         # pre-process distribution
        #         next_tokens_scores = logits_processor(input_ids, next_token_logits)

        #         # Store scores, attentions and hidden_states when required
        #         if return_dict_in_generate:
        #             if output_scores:
        #                 scores += (next_tokens_scores,)
        #             if output_attentions:
        #                 decoder_attentions += (
        #                     (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
        #                 )
        #                 if self.config.is_encoder_decoder:
        #                     cross_attentions += (outputs.cross_attentions,)

        #             if output_hidden_states:
        #                 decoder_hidden_states += (
        #                     (outputs.decoder_hidden_states,)
        #                     if self.config.is_encoder_decoder
        #                     else (outputs.hidden_states,)
        #                 )

        #         # argmax
        #         next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        #         next_tokens_uncombined = torch.argmax(next_token_uncombined_logits, dim=-1)
        #         next_tokens_expert = torch.argmax(next_token_expert_weights)
        #         # print("next_tokens", next_tokens)
        #         # print("next_tokens_uncombined", next_tokens_uncombined)
        #         # print("next_tokens_expert", next_tokens_expert )
        #         # finished sentences should have their next token be a padding token
        #         if eos_token_id is not None:
        #             if pad_token_id is None:
        #                 raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        #             next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        #             next_tokens_uncombined = next_tokens_uncombined[:, None] * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
        #         # update generated ids, model inputs, and length for next step
        #         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        #         uncombined_sequences.append(next_tokens_uncombined)
        #         expert_selections.append(next_tokens_expert)
        #         model_kwargs = self._update_model_kwargs_for_generation(
        #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        #         )
        #         cur_len = cur_len + 1

        #         # if eos_token was found in one sentence, set sentence to finished
        #         if eos_token_id is not None:
        #             unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        #         # stop when each sentence is finished, or if we exceed the maximum length
        #         if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
        #             if not synced_gpus:
        #                 break
        #             else:
        #                 this_peer_finished = True

        #     # if return_dict_in_generate:
        #     #     if self.config.is_encoder_decoder:
        #     #         return GreedySearchEncoderDecoderOutput(
        #     #             sequences=input_ids,
        #     #             scores=scores,
        #     #             encoder_attentions=encoder_attentions,
        #     #             encoder_hidden_states=encoder_hidden_states,
        #     #             decoder_attentions=decoder_attentions,
        #     #             cross_attentions=cross_attentions,
        #     #             decoder_hidden_states=decoder_hidden_states,
        #     #         )
        #     #     else:
        #     #         return GreedySearchDecoderOnlyOutput(
        #     #             sequences=input_ids,
        #     #             scores=scores,
        #     #             attentions=decoder_attentions,
        #     #             hidden_states=decoder_hidden_states,
        #     #         )
        #     # else:
        #     expert_selections = torch.stack(expert_selections)
        #     uncombined_sequences = torch.stack(uncombined_sequences)
        #     uncombined_sequences = uncombined_sequences.reshape(uncombined_sequences.shape[1], uncombined_sequences.shape[0], -1)
        #     # print("the whole sequence", input_ids, expert_selections, uncombined_sequences)
        #     return input_ids, expert_selections, uncombined_sequences 


    
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



# class KnowledgeSelection(torch.nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.hc_proj = nn.Linear(embed_dim, 1)
#         self.hd_proj = nn.Linear(embed_dim, 1)
    
#     def forward(self, lm_logits, encoder_hidden, decoder_hidden, n_expert):
#         '''
#         p.size() = (B x N_expert, L)
#         '''
#         # encoder-decoder interaction
#         A = torch.bmm(decoder_hidden, encoder_hidden.transpose(1,2)) # (B, Ld, Le*N)
#         A = nn.functional.softmax(A, dim=-1)
#         Hc = torch.bmm(A, encoder_hidden) # encoder_hidden = (B, Le, H); Hc = (B, Ld, H)
#         p = self.hc_proj(Hc) + self.hd_proj(decoder_hidden) # (B, Ld, 1)
        
#         p = p.view(-1, n_expert, p.size(-2), p.size(-1))
#         p = nn.functional.softmax(p, dim=1)
#         p = p.view(-1, p.size(-2), p.size(-1))
#         # print("Expert weight", p)
#         # lm_logits.sise = (B x N_expert, L, V)
#         output_logits = lm_logits * p
        

#         return output_logits, p


class KnowledgeSelection(torch.nn.Module):
    '''
    Hard weighting based on similarity.
    '''
    def __init__(self, embed_dim):
        super().__init__()
        # TODO add hidden layers
    
    def forward(self, lm_logits, encoder_hidden, decoder_hidden, n_expert, similarity, hard_weight):

        Ld = decoder_hidden.shape[1] # decoder sequence length
        Bsz = int(decoder_hidden.shape[0] / n_expert)
        # similarity: (B x N, 1)
        _p = similarity.reshape(-1, n_expert, 1) # (B, N, 1)

        # Give Q only the average of the other four
        p = _p.clone()
        p[:, 0, :] = torch.mean(_p[:, 1:, :], dim=1)
        
        if hard_weight:
            max_indices = torch.argmax(p, dim=1, keepdim=True)
            p = torch.zeros_like(p)
            p.scatter_(1, max_indices, 1)
        else:
            p = F.softmax(p, dim=1)
        p = p.reshape(-1, 1, 1) # (B*N, 1, 1)
        p = p.expand(-1, Ld, 1).clone().to(lm_logits.device)
        output_logits = lm_logits * p

        

        return output_logits, p

