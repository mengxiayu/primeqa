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
    # triple_set = set()
    for text in texts:
        resolved_text = nlp(text)._.coref_resolved
        doc = client.annotate(resolved_text)
        # print(resolved_text)
        for sent in doc.sentence:
            # https://github.com/stanfordnlp/stanza/blob/011b6c4831a614439c599fd163a9b40b7c225566/doc/CoreNLP.proto#L558
            for triple in sent.openieTriple:
                score = triple.confidence
                # rel = triple.relation
                if score < 0.8:
                    continue

                new_str = f"{triple.subject}\t{triple.object}"
                str_list.append(new_str)

    str_list = list(set(str_list))
    return str_list



# def return_entities(sent: str):
#     return [
#         token.lemma_.lower() for token in nlp(sent)
#         if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
#     ]



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

def load_asqa_dpr(dataset_fn, n_docs):
    import json
    examples = []
    with open(dataset_fn) as f:
        for line in f:
            ex = {}
            data = json.loads(line.strip())
            ex["id"] = data["id"]
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

def create_external_graph(task_path, examples, start=None, end=None, port=None):
    print("start and end", start, end)
    examples = examples[start:end]
    fw = open(task_path + f"id2kg_{start}_{end}.json", 'w')
    with CoreNLPClient(annotators=['openie'], 
                   memory='16G', endpoint=port, be_quiet=True, properties=properties) as client:
        for ex in tqdm(examples, total=len(examples)):
            
            kg_docs_text = ex["passages"] # a list of strings
            str_list = convert2kg(kg_docs_text, client)
            new_line = {
                "id": ex["id"],
                "kg": str_list
            }
            fw.write(json.dumps(new_line)+'\n')
            
    fw.close()


if __name__ == '__main__':
    import sys
    args = sys.argv
    assert len(args) == 6

    examples = load_asqa_dpr(
        args[3],
        n_docs=20
    )   
    output_path = args[4]
    assert os.path.isdir(output_path)
    create_external_graph(output_path, examples, int(args[1]), int(args[2]), args[5])
