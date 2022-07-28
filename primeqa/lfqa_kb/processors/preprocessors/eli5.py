from typing import List, Tuple
import torch

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])


### For CMSKS 
def preprocess_eli5_function_cmsks(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indices, inputs, targets, queries = preprocess_eli5_batch_cmsks(examples, data_args, mode="train")
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
    model_inputs["example_id"] = indices
    model_inputs["query"] = queries
    return model_inputs

def preprocess_eli5_validation_function_cmsks(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indexes, inputs, targets, queries = preprocess_eli5_batch_cmsks(examples, data_args, mode="eval")
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

def preprocess_eli5_batch_cmsks(examples, data_args, mode="train") -> Tuple[List[str], List[str]]:
    
    questions = examples[data_args.question_column]
    answers = examples[data_args.answer_column]
    contexts = examples[data_args.context_column]
    n_doc = data_args.n_context

    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]
    def append_question(passages, question):
        return [f"question: {question} passage: {t}" for t in passages]
    def get_initial_query(question):
        query = [
            token.lemma_.lower() for token in nlp(question)
            if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
        ]
        return list(set(query))
    # multiple answers for training
    if mode == "train":
        indices = []
        inputs = []
        targets = []
        queries = [] # initial query for kg. It is a list of words made from the question
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            answer_list = answers[idx]
            if len(answer_list) == 0:
                inputs.append(question_passages)
                targets.append("")  
                indices.append(examples["id"][idx])
            else: # multiple answers
                for answer_data in answer_list:
                    a = answer_data["answer"]
                    answer_score = answer_data["meta"]["score"]     
                    if answer_score >= 3: # only takes answers whose score>3
                        inputs.append(question_passages)
                        targets.append(a)
                        indices.append(examples["id"][idx])
                        queries.append(get_initial_query(q))
                    
    elif mode == "eval": # for evaluation only take each question once
        inputs = []
        indices = []
        queries = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            inputs.append(question_passages)
            indices.append(examples["id"][idx])
            queries.append(get_initial_query(q))
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")

    return indices, inputs, targets, queries # inputs is a list of a list of question+passage, targets is a list of answers



### For CMSKS oracle ###
def preprocess_eli5_function_cmsks_oracle(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indices, inputs, targets, queries = preprocess_eli5_batch_cmsks_oracle(examples, data_args, mode="train")
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
    model_inputs["example_id"] = indices
    model_inputs["query"] = queries
    return model_inputs

def preprocess_eli5_validation_function_cmsks_oracle(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indices, inputs, targets, queries = preprocess_eli5_batch_cmsks_oracle(examples, data_args, mode="eval")
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



def preprocess_eli5_batch_cmsks_oracle(examples, data_args, mode="train") -> Tuple[List[str], List[str]]:
    '''
    queries are oracle KG words: answer words
    '''
    questions = examples[data_args.question_column]
    answers = examples[data_args.answer_column]
    contexts = examples[data_args.context_column]
    n_doc = data_args.n_context

    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]
    def append_question(passages, question):
        return [f"question: {question} passage: {t}" for t in passages]
    def get_initial_query(x):
        y = [
            token.lemma_ for token in nlp(x) if
            not token.is_stop
            and not token.is_currency
            and not token.is_digit
            and not token.is_punct
            and not token.is_space
            and not token.like_num
            and not token.pos_ == "PROPN"
        ]
        return list(set(y))
    # multiple answers for training
    if mode == "train":
        indices = []
        inputs = []
        targets = []
        queries = [] # initial query for kg. It is a list of words made from the question
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            answer_list = answers[idx]
            if len(answer_list) == 0:
                inputs.append(question_passages)
                targets.append("")  
                indices.append(examples["id"][idx])
            else: # multiple answers
                for answer_data in answer_list:
                    a = answer_data["answer"]
                    answer_score = answer_data["meta"]["score"]     
                    if answer_score >= 3: # only takes answers whose score>3
                        inputs.append(question_passages)
                        targets.append(a)
                        indices.append(examples["id"][idx])
                        queries.append(get_initial_query(a)) # use answer to query
                        # print(get_initial_query(a))
                    
    elif mode == "eval": # for evaluation only take each question once
        inputs = []
        indices = []
        queries = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            inputs.append(question_passages)
            indices.append(examples["id"][idx])
            answer_list = answers[idx]
            answer_texts = [answer_data["answer"] if answer_data["answer"] else "" for answer_data in answer_list]
            qry = get_initial_query(" ".join(answer_texts))
            print(examples["id"][idx], qry)
            queries.append(qry)
            
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")

    return indices, inputs, targets, queries # inputs is a list of a list of question+passage, targets is a list of answers



#### For FiD ####
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
    indexes, inputs, targets = preprocess_eli5_batch_fid(examples, data_args, mode="train")
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
    return model_inputs

def preprocess_eli5_validation_function_fid(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    indexes, inputs, targets = preprocess_eli5_batch_fid(examples, data_args, mode="eval")
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

    return model_inputs

def preprocess_eli5_batch_fid(examples, data_args, mode="train") -> Tuple[List[str], List[str]]:
    indices = []
    questions = examples[data_args.question_column]
    answers = examples[data_args.answer_column]
    contexts = examples[data_args.context_column]
    n_doc = data_args.n_context

    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]
    def append_question(passages, question):
        return [f"question: {question} passage: {t}" for t in passages]
    # multiple answers for training
    if mode == "train":
        inputs = []
        targets = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            answer_list = answers[idx]
            if len(answer_list) == 0:
                inputs.append(question_passages)
                targets.append("")  
                indices.append(examples["id"][idx])
            else: # multiple answers
                for answer_data in answer_list:
                    a = answer_data["answer"]
                    answer_score = answer_data["meta"]["score"]     
                    if answer_score >= 3: # only takes answers whose score>3
                        inputs.append(question_passages)
                        targets.append(a)
                        indices.append(examples["id"][idx])
                    
    elif mode == "eval": # for evaluation only take each question once
        inputs = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            question_passages = append_question(passages, q)
            inputs.append(question_passages)
            indices.append(examples["id"][idx])
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")

    return indices, inputs, targets # inputs is a list of a list of question+passage, targets is a list of answers



#### For BART ####


def preprocess_eli5_validation_function(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    inputs, targets = preprocess_eli5_batch(examples, data_args, mode="eval")

    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding,
        truncation=True,
        # return_overflowing_tokens=True, # See this issue https://github.com/huggingface/transformers/issues/15398#issuecomment-1072714545
        return_offsets_mapping=True,
    )
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

    model_inputs["example_id"] = []

    for i in range(len(model_inputs["input_ids"])):
        model_inputs["example_id"].append(examples["id"][i])

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# for Train set. tokenize.
def preprocess_eli5_function(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    # if data_args.n_context > 1 and data_args.train_passage_file is not None:
    #     inputs, targets, passages = preprocess_eli5_batch(examples, question_column, answer_column, mode="train", passage_file=data_args.train_passage_file, n_doc=data_args.n_context)
    # else:
    inputs, targets = preprocess_eli5_batch(examples, data_args, mode="train")
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs # "labels", "input_ids", "attention_mask".


def preprocess_eli5_batch(examples, data_args, mode="train") -> Tuple[List[str], List[str]]:
    # return inputs, targets texts
    # concatenate contexts with the question if they exist
    questions = examples[data_args.question_column]
    answers = examples[data_args.answer_column]
    n_doc = data_args.n_context
    with_passage = data_args.context_column is not None and n_doc > 0
    if with_passage:    
        contexts = examples[data_args.context_column]

    def generate_input(_question):
        return " ".join(["question:", _question.lstrip()])
    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]
    def concat_question_passages(passages, question):
        passages_concatenation = " ".join(passages)
        return f"question: {question} passage: {passages_concatenation}"

    # multiple answers for training
    if mode == "train":
        inputs = []
        targets = []
        for idx,question in enumerate(questions):
            if with_passage:
                passages = top_passages(contexts[idx])
                q = concat_question_passages(passages, question)
            else:  
                q = generate_input(question)
            answer_list = answers[idx]
            if len(answer_list) == 0:
                inputs.append(q)
                targets.append("")  
            else: # multiple answers
                for answer_data in answer_list:
                    a = answer_data["answer"]
                    answer_score = answer_data["meta"]["score"]     
                    if answer_score >= 3: # only takes answers whose score>3
                        inputs.append(q)
                        targets.append(a)
                    
    elif mode == "eval": # for evaluation only take each question once
        if with_passage:
            inputs = [concat_question_passages(top_passages(contexts[idx]), questions[idx]) for idx in range(len(questions))]
        else:
            inputs = [generate_input(question) for question in questions]
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")
    return inputs, targets


