from typing import List, Tuple
import torch

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])


#### For FiD + n-hop KG ####
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
    if data_args.kg_column is not None:
        vocabs = examples[data_args.kg_column]
    n_doc = data_args.n_context

    def top_passages(ctx):
        assert n_doc <= len(ctx) 
        return [ctx[i]["text"] for i in range(n_doc)]

    def get_top_answers(answers):
        # sort answers by recall then take top n
        answers_with_recall = {}
        top_answers = []
        for answer in answers:
            if mode == 'train' and answer['meta']['score'] < 3:
                continue
            # if answer['meta'] is None or 'recall' not in answer['meta'] or answer['meta']['recall'] == None:
            #      continue
            if data_args.keep_top_n_answer is not None:
                answers_with_recall[answer['answer']] = answer['meta']['recall']
            else:
                top_answers.append(answer['answer'])
        if data_args.keep_top_n_answer is None:
            return top_answers
        answers_with_recall = dict(sorted(answers_with_recall.items(), key=lambda item: item[1], reverse=True))
        count = 0
        for answer in answers_with_recall:
            count+= 1
            top_answers.append(answer)
            if data_args.keep_top_n_answer is not None and count >= data_args.keep_top_n_answer:
                break
        return top_answers

    def append_question(passages, question, vocab):
        return_data = []
        if data_args.q_only:
            return_data.append("question: " + question)
        # check if kg available
        if data_args.use_kg_oracle and data_args.kg_column != None:
            kg_text = ""
            if data_args.kg_column == "kg_vocab":
                kg_text = " ".join(vocab)
            # have kg sentences be a fourth passage
            elif data_args.kg_column == "kg_sentences" or data_args.kg_column == "kg_triples":
                text_field = "text"
                delim = ". "
                if data_args.kg_column == "kg_sentences":
                    text_field = "sentence"
                    delim = " "
                kg_text = ""
                for sentence in vocab:
                    if data_args.vocab_threshold == None or data_args.vocab_threshold <= 0 or (sentence['count'] >= data_args.vocab_threshold):
                        kg_text += sentence[text_field] + delim
            # put kg first
            if data_args.p_b4_q:
                return_data.append(f"passage: {kg_text} question: {question}")
            else:
                return_data.append(f"question: {question} passage: {kg_text}")
        if data_args.p_b4_q:
            return_data.extend([f"passage: {t} question: {question}" for t in passages])
        else:
            return_data.extend([f"question: {question} passage: {t}" for t in passages])
        return return_data
    # multiple answers for training
    if mode == "train":
        inputs = []
        targets = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            answer_list = get_top_answers(answers[idx])
            kg_vocab = None
            if data_args.kg_column is not None:
                kg_vocab = vocabs[idx]
            
            # if there no answers or no passages, then discard. Filter based on answer list, kg sentences or kg_vocab
            if len(answer_list) == 0 or data_args.kg_column is not None and len(kg_vocab) == 0:
                continue
            # elif data_args.apply_filter and (len(answer_list) == 0 or kg_vocab == None or (kg_vocab != None and len(kg_vocab) == 0) or (data_args.kg_column != None and data_args.vocab_threshold != None and len(kg_vocab) > 0 and 
            #     ((data_args.kg_column == 'kg_sentences' and kg_vocab[0]['count'] < data_args.vocab_threshold)
            #     or (data_args.kg_column == 'kg_sentences' and len(kg_vocab) < data_args.vocab_threshold)))):
            #     continue
            else: # multiple answers
                question_passages = append_question(passages, q, kg_vocab)
                
                for answer_data in answer_list:
                    inputs.append(question_passages)
                    targets.append(answer_data)
                    indices.append(examples["id"][idx])
                    
    elif mode == "eval": # for evaluation only take each question once
        inputs = []
        targets = []
        for idx,q in enumerate(questions):
            passages = top_passages(contexts[idx])
            kg_vocab = None
            if data_args.kg_column is not None:
                kg_vocab = vocabs[idx]
            question_passages = append_question(passages, q, kg_vocab)
            inputs.append(question_passages)
            indices.append(examples["id"][idx])
        targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
    else:
        raise ValueError("mode requires eval or train")

    return indices, inputs, targets # inputs is a list of a list of question+passage, targets is a list of answers


