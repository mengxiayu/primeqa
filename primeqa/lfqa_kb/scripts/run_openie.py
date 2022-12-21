import os
from re import X
from typing import Any, List
import gzip

import pandas as pd
import spacy
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from tqdm import tqdm
import json
from stanza.server import CoreNLPClient
import multiprocessing
from multiprocessing import Process
import datetime

use_coref = False

if use_coref:
    nlp = spacy.load("en_core_web_sm")
    nlp_coref = spacy.load("en_coreference_web_trf")
    nlp.add_pipe("transformer", source=nlp_coref)
    nlp.add_pipe("coref", source=nlp_coref)
    nlp.add_pipe("span_resolver", source=nlp_coref)
    nlp.add_pipe("span_cleaner", source=nlp_coref)

properties = {
    'openie.max_entailments_per_clause': 500,
    'openie.affinity_probability_cap': 2 / 3,
    # 'openie.resolve_coref': 'true',
    # 'annotators': 'coref', 
    # 'coref.algorithm' : 'neural'
}

def is_stop_word(triple, sent):
    for token in triple:
        if sent.token[token.tokenIndex].lemma.lower() not in STOP_WORDS and sent.token[token.tokenIndex].pos != 'PRON':
            return False
    return True

# Define lightweight function for resolving references in text
def resolve_references(doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
            
            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string

def get_subphrase(phrase1, phrase2):
    words1 = set(phrase1.split())
    words2 = set(phrase2.split())

    # words1 is the subphrase
    if len(words1 - words2) == 0:
        return phrase1
    # words2 is the subphrase
    elif len(words2 - words1) == 0:
        return phrase2
    else:
        return None

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

    triples = []

    for pid in texts: #nlp.pipe(texts):
        #resolved_text = resolve_references(text) #._.coref_resolved
        doc = client.annotate(texts[pid])
        sent_i = 0
        for sent in doc.sentence:
            triples_sent = {}
            # https://github.com/stanfordnlp/stanza/blob/011b6c4831a614439c599fd163a9b40b7c225566/doc/CoreNLP.proto#L558
            for triple in sent.openieTriple:
                score = triple.confidence
                # rel = triple.relation
                if score < 0.8:
                    continue

                # filtering steps to add:
                # stop word in subject: NLTK stop words, days, months
                # keep longest triple only (qa_srl format does this? doesn't seem to be good so far...)
                # don't allow more than 5 words in a s/r/o
                # subject != object

                if triple.subject.lower() == triple.object.lower():
                    continue
                if len(triple.subject.split()) > 5 or len(triple.relation.split()) > 5 or len(triple.object.split()) > 5:
                    continue
                if is_stop_word(triple.subjectTokens, sent) or is_stop_word(triple.objectTokens, sent):
                    continue
                
                if triple.subject.lower() not in triples_sent:
                    triples_sent[triple.subject.lower()] = {}
                if triple.object.lower() not in triples_sent[triple.subject.lower()]:
                    triples_sent[triple.subject.lower()][triple.object.lower()] = []
                triples_sent[triple.subject.lower()][triple.object.lower()].append(triple.relation.lower())
            
            # If there are two S,R that all O are the same and the S words overlap, only keep the longer S
            # print(str(datetime.datetime.now()) + " filter triples")
            remove = set()
            i = 0
            for triple_i in triples_sent:
                j = 0
                for triple_j in triples_sent:
                    if j <= i:
                        j += 1
                        continue
                    subphrase = get_subphrase(triple_i, triple_j)
                    if subphrase is not None and triples_sent[triple_i] == triples_sent[triple_j]:
                        remove.add(subphrase)
                        if subphrase == triple_i:
                            break
                    j += 1
                i += 1
            for item in remove:
                triples_sent.pop(item)
            # For each S,R only keep the longest O
            for subject_i in triples_sent:
                keep_objects = []
                for object_i in triples_sent[subject_i]:
                    discard = False
                    
                    for i in range(len(keep_objects)):
                        subphrase = get_subphrase(object_i, keep_objects[i])
                        if subphrase is not None:
                            if subphrase == keep_objects[i]:
                                keep_objects[i] = object_i
                            discard = True
                            #break
                    if not discard:
                        keep_objects.append(object_i)
                objects = list(triples_sent[subject_i].keys())
                for object_i in objects:
                    if object_i not in keep_objects:
                        triples_sent[subject_i].pop(object_i)

            for subject_i in triples_sent:
                for object_i in triples_sent[subject_i]:
                    for relation_i in triples_sent[subject_i][object_i]:
                        new_str = f"{subject_i}\t{relation_i}\t{object_i}\t{pid}-{sent_i}"
                    triples.append(new_str)
            sent_i += 1
        #print(str(datetime.datetime.now()) + " passage processed")                    
    return triples

def load_eli5_dpr(dataset_fn, n_docs):
    import json
    examples = []
    with gzip.open(dataset_fn) as f:
        for line in f:
            ex = {}
            data = json.loads(line.strip())
            ex["id"] = data["id"]
            ex["input"] = data["input"]
            ex["passages"] = {}
            seen_pids = dict()
            for i in range(min(n_docs,len(data["passages"]))):
                passage = data["passages"][i]["text"]
                pid = data["passages"][i]['pid']
                offsets = pid[pid.index("::")+2:]
                pid = pid[:pid.index("::")]
                start, end = offsets.split(",") 
                # if start[0] == "(":
                #     start = int(start[1:])+1
                # else:
                start = int(start[1:])
                end = int(end[:-1])

                # if overlapping paragraphs only take the first (highest score)
                if pid in seen_pids:
                    overlap = False
                    for j in range(len(seen_pids[pid]['start'])):
                        if (start >= seen_pids[pid]['start'][j] and start <= seen_pids[pid]['end'][j]) or \
                            (end >= seen_pids[pid]['start'][j] and end <= seen_pids[pid]['end'][j]) or \
                                (start <= seen_pids[pid]['start'][j] and end >= seen_pids[pid]['end'][j]):
                                overlap = True
                    if not overlap:
                        seen_pids[pid]['start'].append(start)
                        seen_pids[pid]['end'].append(end)
                    else:
                        continue
                else:
                    seen_pids[pid] = {}
                    seen_pids[pid]['start'] = [start]
                    seen_pids[pid]['end'] = [end]

                
                # add period after title so it doesn't confuse openIE
                if start == 0:
                    ex["passages"][data["passages"][i]['pid']] = data["passages"][i]["title"] + "." + passage[len(data["passages"][i]["title"]):]
                else:
                    ex["passages"][data["passages"][i]['pid']] = passage
            # ex["passages"] =  [x["text"] for x in data["passages"]][:n_docs]
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

os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'

def create_external_graph(task_path, examples, start=None, end=None, port=9000):
    print(str(datetime.datetime.now()) + "start and end: ", start, end)
    #print(datetime.datetime.now())

    if start is not None and end is not None:
        examples = examples[start:end]
    fw = open(task_path, 'w')
    with CoreNLPClient(annotators=['openie'], # tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref',
                   memory='16G', be_quiet=True, properties=properties, endpoint="http://localhost:" + str(port)) as client:
        print(str(datetime.datetime.now()) + " started corenlp")
        for ex in tqdm(examples, total=len(examples)):
            kg_docs_text = ex["passages"] # a list of strings
            #proc = Process(target=convert2kg, args=(kg_docs_text, client))
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
    #assert len(args) == 6

    dataset_fn = args[1]
    output_path = args[2]
    port = args[3]
    # start = int(args[1])
    # end = int(args[2])

    # port = None #'http://localhost:9000' #args[5]

    examples = load_eli5_dpr(
        dataset_fn,
        n_docs=100
    )   
    
    # assert os.path.isdir(output_path)
    create_external_graph(output_path, examples, port=port) #, int(start), int(end))
