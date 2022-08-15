from typing import List, Tuple
import torch

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])


#### For FiD KG ####
# from 
# https://github.com/facebookresearch/FiD/blob/25ed1ff0fe0288b80fb5e9e5de8d6346b94b8d48/src/data.py#L73
def encode_passages(batch_text_passages, tokenizer, max_length):
    '''
    Param: 
        batch_text_passages: (bsz, n_doc, )
    '''
    passage_ids, passage_masks = [], []
    for text_passages in batch_text_passages:
        # p = tokenizer.batch_encode_plus(
        p = tokenizer(
            text_passages,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids.tolist(), passage_masks.tolist()

def preprocess_eli5_function_fid(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indexes, inputs, targets, queries = preprocess_eli5_batch_fid(examples, data_args, mode="train")
    passage_ids, passage_masks = encode_passages(inputs, tokenizer, max_seq_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs = {}
    model_inputs["input_ids"] = passage_ids
    model_inputs["attention_mask"] = passage_masks
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["example_id"] = indexes
    model_inputs["query"] = queries
    return model_inputs

def preprocess_eli5_validation_function_fid(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indexes, inputs, targets, queries = preprocess_eli5_batch_fid(examples, data_args, mode="eval")
    passage_ids, passage_masks = encode_passages(inputs, tokenizer, max_seq_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs = {}
    model_inputs["input_ids"] = passage_ids
    model_inputs["attention_mask"] = passage_masks
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["example_id"] = []
    for i in range(len(model_inputs["input_ids"])):
        model_inputs["example_id"].append(examples["id"][i])
    model_inputs["query"] = queries
    return model_inputs

def preprocess_eli5_batch_fid(examples, data_args, mode="train") -> Tuple[List[str], List[str]]:
    indices = []
    questions = examples[data_args.question_column]
    answers = examples[data_args.answer_column]
    contexts = examples[data_args.context_column]
    vocabs = examples["kg_vocab"]
    n_doc = data_args.n_context

    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]
    def append_question(passages, question, vocab):
        vocab_text = " ".join(vocab)
        result = [f"question: {question} passage: {vocab_text}"] + [f"question: {question} passage: {t}" for t in passages]
        return result
    def get_initial_query(question):
        query = [
            token.lemma_.lower() for token in nlp(question)
            if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
        ]
        return list(set(query))
    
    if mode == "train":
        inputs = []
        targets = []
        queries = [] # initial query for kg. It is a list of words made from the question
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            answer_list = answers[idx]
            kg_vocab = vocabs[idx]
            if len(answer_list) == 0:
                inputs.append(append_question(passages, q, kg_vocab))
                targets.append("")  
                indices.append(examples["id"][idx])
            else: # multiple answers
                for answer_data in answer_list:
                    a = answer_data["answer"]
                    answer_score = answer_data["meta"]["score"]     
                    if answer_score >= 3: # only takes answers whose score>3
                        question_passages = append_question(passages, q, kg_vocab)
                        inputs.append(question_passages)
                        targets.append(a)
                        indices.append(examples["id"][idx])
                        queries.append(get_initial_query(q))
                    
    elif mode == "eval": # for evaluation only take each question once
        inputs = []
        queries = [] # initial query for kg. It is a list of words made from the question
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            kg_vocab = vocabs[idx]
            question_passages = append_question(passages, q, kg_vocab)
            inputs.append(question_passages)
            indices.append(examples["id"][idx])
            queries.append(get_initial_query(q))
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")

    return indices, inputs, targets, queries # inputs is a list of a list of question+passage, targets is a list of answers

