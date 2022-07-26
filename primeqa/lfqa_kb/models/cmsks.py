# from data collator (index, target_ids, target_mask, passage_ids, passage_masks)

import tokenizers
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn, softmax
import random
import json
import numpy as np
import transformers
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, Seq2SeqLMOutput
import marisa_trie
from typing import Optional, Dict, Any
import pickle

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
from transformers.models.bart.modeling_bart import logger, shift_tokens_right

class FiDBART(transformers.BartForConditionalGeneration):

    def __init__(self, config, kg_file=None): # TODO pass tokenizer
        super().__init__(config)
        self.embed_dim = config.d_model
        self.wrap_encoder()
        self.knowledge_trie = self.load_external_kg(kg_file)
        self.knowledge_selection = KnowledgeSelection(self.embed_dim)

    def load_external_kg(self, kg_file):
        # with open(kg_file, 'r') as f:
        #     return json.loads(f.read())
        return pickle.load(open(kg_file, "rb"))
    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(kwargs["input_ids"].size(0), -1)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].size(0), -1)

        return super(FiDBART, self).forward(
            **kwargs
        )    
    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=False, example_id=None, query=None, decoder_inputs_embeds=None, use_cache=None, decoder_input_ids=None, **kwargs):
        if input_ids != None:
            if input_ids.dim() == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1) # reshape input_ids from 3 dim to 2 dim
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)


        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        kg_logits = self.calculate_knowledge_dist(
            lm_logits=lm_logits,
            max_hops=3,
            example_ids=example_id,
            query=query,
            )

        # Option 1: equally select
        # final_logits = lm_logits + kg_logits

        # Option 2: select by learned weight TODO
        if not return_dict:
            decoder_hidden, encoder_hidden = outputs
        else:
            decoder_hidden = outputs.last_hidden_state
            encoder_hidden = outputs.encoder_last_hidden_state
        final_logits = self.knowledge_selection(lm_logits, kg_logits, encoder_hidden, decoder_hidden)
        # output
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(final_logits.view(-1, self.config.vocab_size), labels.view(-1))


        if not return_dict:
            output = (final_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
       
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=final_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )        
        
    
    def calculate_knowledge_dist(self, example_ids=None, lm_logits=None, query=None, max_hops=None):
        '''
        query from the knowledge trie (two hops)
        locate the tokens to vodabulary
        modify the lm_logits 
        '''
        indicator = torch.zeros(lm_logits.shape)
        for idx,exp_id in enumerate(example_ids):
            # str_list = self.knowledge_trie[exp_id] # TODO pass it from the dataset
            # ext_trie = marisa_trie.Trie(str_list)
            ext_trie = self.knowledge_trie[exp_id]
            local_kg = query[idx] # a list of tokens
            tmp_kg = local_kg
            related_kgs = set(local_kg)
            for i in range(max_hops):
                new_knowledge = []
                for ent in tmp_kg:
                    for span in ext_trie.keys(ent):
                        new_knowledge.extend(span.split(' '))
                new_knowledge = set(new_knowledge)
                tmp_kg = list(new_knowledge)
                related_kgs |= new_knowledge # this is the kg vocab
            token_ids = tokenizer(' '.join(list(related_kgs)))["input_ids"]
            indicator[idx, :, token_ids] = 1
        indicator = indicator.to(lm_logits.device)
        kg_logits = lm_logits * indicator
            
        return kg_logits

    # customize this function to allow "query" and "example_id"
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        query=None,
        example_id=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "query": query,
            "example_id": example_id
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "query", "example_id"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs
    
    def generate(self, input_ids, **gen_kwargs):
        self.model.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids,
            **gen_kwargs
        )
    
    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap encoder to obtain an FiD model
        """
        self.model.encoder = EncoderWrapper(self.model.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap FiD, useful to load weights
        """
        self.model.encoder = self.model.encoder.encoder 
    
    def set_checkpoint(self, use_checkpoint):
        """
        deprecated
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.model.encoder.encoder.layers:
            mod.use_checkpoint = use_checkpoint
    
    # only used when initialized from pretrained BART. 
    # BART doesn't contain knowledge_selection module, so we initialize it here.
    def load_pretrained(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False) # strict=False to allow partial load.
        self.wrap_encoder()
        
    
class KnowledgeSelection(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hc_proj = nn.Linear(embed_dim, 1)
        self.hd_proj = nn.Linear(embed_dim, 1)
    
    def forward(self, lm_logits, kg_logits, encoder_hidden, decoder_hidden):
        '''
        A = softmax(hdec hTenc)
        hc = Ahenc
        pgen = sigmod(Wc hc + Wg hdec )
        '''
        A = torch.bmm(decoder_hidden, encoder_hidden.transpose(1,2)) # (B, Ld, Le)
        A = nn.functional.softmax(A, dim=-1)
        Hc = torch.bmm(A, encoder_hidden) # encoder_hidden = (B, Le, N); Hc = (B, Ld, N)
        p = torch.sigmoid(self.hc_proj(Hc) + self.hd_proj(decoder_hidden))

        return p * lm_logits + (1-p) * kg_logits
        


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input_ids=None, attention_mask=None, return_dict=False, **kwargs):
        # total_length = n_passages * passage_length
        if input_ids.dim() == 3: # the generate() function directly call the encoder, so we don't have chance to resize before encoder
            input_ids = input_ids.view(input_ids.size(0), -1)
        bsz, total_length = input_ids.shape # B * (N * L)
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length) # resize to (B * N) * L
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        if not return_dict:
            return (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:] # concatenate encoder outputs 

        return BaseModelOutput( # TODO pass hidden_states and attentions
            last_hidden_state=outputs[0].view(bsz, self.n_passages*passage_length, -1),
        )
