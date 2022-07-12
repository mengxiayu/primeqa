# from data collator (index, target_ids, target_mask, passage_ids, passage_masks)

import tokenizers
import torch
import torch.nn.functional as F
from torch import nn
import random
import json
import numpy as np
import transformers
from transformers.modeling_outputs import BaseModelOutput
import marisa_trie

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

class FiDBART(transformers.BartForConditionalGeneration):

    def __init__(self, config, kg_file=None):
        super().__init__(config)
        self.wrap_encoder()
        self.knowledge_trie = self.load_external_kg(kg_file)
    def load_external_kg(self, kg_file):
        with open(kg_file, 'r') as f:
            return json.loads(f.read())
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
    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=False, example_id=None, query=None, **kwargs):
        if input_ids != None:
            if input_ids.dim() == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        ) # tuple: (loss, lm_logits,) + BartModel outputs (all models return loss in the first element)
        # output[1] shape: (bsz, l, vocab_size)
        
        kg_outputs = self.calculate_knowledge_dist(
            lm_logits=outputs[1],
            max_hops=2,
            example_ids=example_id,
            query=query,
            )

        '''
        TODO modify here. recalculate the lm_logits and the loss.
        '''
        return outputs

    def calculate_knowledge_dist(self, example_ids=None, lm_logits=None, query=None, max_hops=None):
        '''
        query from the knowledge trie (two hops)
        locate the tokens to vodabulary
        modify the lm_logits 
        '''
        batch_relatd_kgs = []
        for idx,exp_id in enumerate(example_ids):
            str_list = self.knowledge_trie[exp_id]
            ext_trie = marisa_trie.Trie(str_list)
            local_kg = query[idx] # a list of tokens
            tmp_kg = local_kg
            related_kgs = set()
            for i in range(max_hops):
                new_knowledge = []
                for ent in tmp_kg:
                    for span in ext_trie.keys(ent):
                        new_knowledge.extend(span.split(' '))
                new_knowledge = set(new_knowledge)
            tmp_kg = list(new_knowledge)
            related_kgs |= new_knowledge # this is the kg vocab
            batch_relatd_kgs.append(' '.join(list(related_kgs)))
        
        token_ids = tokenizer(batch_relatd_kgs)["input_ids"]
        print(token_ids.shape)
        indicator = torch.zeros(lm_logits.shape)
        indicator[token_ids[0],:, token_ids[1]] = 1
        indicator = indicator.to(lm_logits.device)
        kg_logits = lm_logits * indicator
            
        return kg_logits

    
    
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
        # TODO the original code assign layers, do we need to do this?
        # layers = []
        # for mod in self.model.encoder.layers:
        #     layers.append(mod.modules())
        # layers = nn.ModuleList(layers)
        # self.model.encoder.layers = layers

    def load_pretrained(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
    
    def set_checkpoint(self, use_checkpoint):
        """
        deprecated
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.model.encoder.encoder.layers:
            mod.use_checkpoint = use_checkpoint
    
    # it was load_t5 in the original repo, we modify it to BART
    def load_pretrained(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
    



class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input_ids=None, attention_mask=None, return_dict=False, **kwargs):
        # total_length = n_passages * passage_length
        if input_ids.dim() == 3: # the generate() function directly call the encoder, so we don't have chance to resize before encoder TODO
            input_ids = input_ids.view(input_ids.size(0), -1)
        bsz, total_length = input_ids.shape # B * (N * L)
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length) # resize to (B * N) * L
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        if not return_dict:
            return (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:] # concatenate encoder outputs #TODO support when return_dict=True

        return BaseModelOutput( # TODO pass hidden_states and attentions
            last_hidden_state=outputs[0].view(bsz, self.n_passages*passage_length, -1),
        )



        
    
