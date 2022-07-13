import os
import pickle
from re import X
from typing import Any, List

import marisa_trie
import neuralcoref
# import pandas as pd
import spacy
import torch
# from datasets import load_dataset
# from openie import StanfordOpenIE
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from tqdm import tqdm
# from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

# import KID
 
# ipywidgets

stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS

Token.set_extension('is_stop', getter=stop_words_getter, force=True)

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)

neuralcoref.add_to_pipe(nlp)

properties = {
    'openie.max_entailments_per_clause': 500,
    'openie.affinity_probability_cap': 2 / 3,
}

# retriever_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained(
#     "facebook/rag-token-nq", index_name="legacy", use_dummy_dataset=False
# )
# rag_model = RagTokenForGeneration.from_pretrained(
#     "facebook/rag-token-nq", retriever=retriever
# )

# kg_dataset = load_dataset("wiki_dpr", 'psgs_w100.nq.exact')


# def return_retrieved_psgs(
#     tokenizer: Any, retriever: Any, rag: Any, query: str, k_docs: int
# ):
#     """Retrieve docs by query.

#     :param tokenizer: the tokenizer of RAG
#     :param rag: the RAG model loaded
#     :param retriever: the retriever of RAG
#     :param query: the question/context of the NLG task. A string.
#     :param k_docs: the number of documents to be retrieved
#     :return: two lists. [doc ids] and [doc scores].
#     """
#     inputs = tokenizer(query, return_tensors="pt")
#     input_ids = inputs["input_ids"]

#     question_hidden_states = rag.question_encoder(input_ids)[0]
#     docs_dict = retriever(
#         input_ids.numpy(),
#         question_hidden_states.detach().numpy(),
#         return_tensors="pt",
#         n_docs=k_docs
#     )
#     doc_scores = torch.bmm(
#         question_hidden_states.unsqueeze(1),
#         docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
#     ).squeeze(1)
#     return docs_dict['doc_ids'].detach().cpu().numpy().tolist()[0], doc_scores.detach(
#     ).cpu().numpy().tolist()[0]


def convert2kg(texts: List[str], client: Any):
    """Extract <subj., rel., obj.> triplets from list of texts.

    :param texts: a list of text
    :param client: the CoreNLPClient openie client
    :return:
    """
    # str_list = []
    # for text in texts:
    #     resolved_text = nlp(text)._.coref_resolved
    #     for triple in client.annotate(resolved_text):
    #         str_list.append(triple['subject'] + ' ' + triple['object'])
    # return str_list

    str_list = []
    triple_set = set()
    for text in texts:
        resolved_text = nlp(text)._.coref_resolved
        doc = client.annotate(resolved_text)
        # print(resolved_text)
        for sent in doc.sentence:
            # https://github.com/stanfordnlp/stanza/blob/011b6c4831a614439c599fd163a9b40b7c225566/doc/CoreNLP.proto#L558
            for triple in sent.openieTriple:
                score = triple.confidence
                # rel = triple.relation
                # print(s, score)
                # s = ' '.join(return_entities(s)) # FIXME The original KID code doesn't do preprocessing but the paper mentioned (lower, preprocessing, etc) 
                if score < 0.8:
                    continue
                    
                subj_tokens = [token.text for token in nlp(triple.subject) if not token.is_stop]
                obj_tokens = [token.text for token in nlp(triple.object) if not token.is_stop]
                if len(subj_tokens)==0 or len(obj_tokens)==0:
                    continue
                # subj + obj
                new_list = []
                for tok in subj_tokens + obj_tokens:
                    if tok not in new_list:
                        new_list.append(tok)
                
                if len(new_list) <= 1:
                    continue
                new_str = " ".join(new_list)
                if new_str not in triple_set:
                    triple_set.add(new_str)
                    str_list.append(new_str)
                # obj + subj
                new_list = []
                for tok in obj_tokens + subj_tokens:
                    if tok not in new_list:
                        new_list.append(tok)
                new_str = " ".join(new_list)
                if new_str not in triple_set:
                    triple_set.add(new_str)
                    str_list.append(new_str)
    # TODO filter str_list

    return str_list



def return_entities(sent: str):
    return [
        token.lemma_.lower() for token in nlp(sent)
        if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
    ]



def load_eli5_dpr(dataset_fn, n_docs):
    import json
    examples = []
    with open(dataset_fn) as f:
        for line in f:
            ex = {}
            data = json.loads(line.strip())
            ex["id"] = data["id"]
            ex["input"] = data["input"]
            ex["passages"] =  [x["text"] for x in data["passages"]][:n_docs]
            examples.append(ex)
    return examples

def load_asqa_content(dataset_fn):
    import json
    examples = []
    with open (dataset_fn) as f:
        for line in f:
            ex = {}
            data = json.loads(line.strip())
            ex["id"] = data["id"]
            ex["input"] = data["ambiguous_question"]
            ex["passages"] = []
            for a in data["annotations"]:
                for x in a["knowledge"]:
                    ex["passages"].append(x["content"])
            examples.append(ex)
    return examples

import os
import json
from stanza.server import CoreNLPClient
os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'

def create_external_graph_eli5(n_docs, task_path, examples, start=None, end=None, port=None):
    print("start and end", start, end)
    examples = examples[start:end]
    # else take all examples
    # full_str_list = []
    id2kg = {}
    cnt = 0
    with CoreNLPClient(annotators=['openie'], 
                   memory='8G', endpoint=port, be_quiet=True, properties=properties) as client:
        for ex in tqdm(examples, total=len(examples)):
            
            kg_docs_text = ex["passages"] # a list of strings
            str_list = convert2kg(kg_docs_text, client)
            # full_str_list.extend(str_list)
            cnt += 1
            id2kg[ex["id"]] = str_list
            with open(task_path + f"id2kg_{start}_{end}.json", 'w') as fw:
                fw.write(json.dumps(id2kg))
            # with open (task_path + f'/full_str_{n_docs}_{cnt}.txt', 'w') as fw:
            #     for x,y,score in str_list:
            #         fw.write(f"{x}\t{y}\t{score}\n")

    # trie = marisa_trie.Trie(full_str_list)
    # with open(task_path + '/trie_' + str(n) + '.pickle', 'wb') as handle:
    #     pickle.dump(trie, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_external_graph_asqa(n_docs, task_path, examples, start=None, end=None, port=None):
    print("start and end", start, end)
    examples = examples[start:end]
    # else take all examples
    full_str_list = []
    with CoreNLPClient(annotators=['openie'], 
                   memory='8G', endpoint=port, be_quiet=True) as client:
        for ex in tqdm(examples, total=len(examples)):
            kg_docs_text = ex["passages"] # a list of strings
            
            full_str_list.extend(convert2kg(kg_docs_text, client))

    with open (task_path + f'/full_str_{n_docs}_{start}-{end}.txt', 'w') as fw:
        for subj, obj, score in full_str_list:
            fw.write(f"{subj}\t{obj}\t{score}\n")


if __name__ == '__main__':

    n=20

    
    import sys
    args = sys.argv
    assert len(args) == 6

    examples = load_eli5_dpr(
        args[3],
        n_docs=n
    )   
    output_path = args[4]
    assert os.path.isdir(output_path)
    create_external_graph_eli5(n, output_path, examples, int(args[1]), int(args[2]), args[5])

    # examples = load_asqa_content(
    #     "/dccstor/myu/data/asqa/asqa_dev.json",
    # )
    # output_path = "/dccstor/myu/experiments/knowledge_trie/asqa_dev"
    # create_external_graph_asqa(n, output_path, examples)