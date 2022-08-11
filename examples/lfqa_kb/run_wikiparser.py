import json
import spacy
import requests
import json


# load passages from dev
def load_passage_sentences(data_fn, cnt):
    ids = []
    passages = []
    with open(data_fn, 'r') as f:
        while cnt > 0:
            example = json.loads(f.readline().strip())
            ids.append(example["id"])
            passages.append([x["text"] for x in example["passages"]])
            cnt -= 1
    return ids, passages


# import neuralcoref
nlp = spacy.load("en_core_web_sm")
# neuralcoref.add_to_pipe(nlp)
def docs2sents(texts):
    sentences = []
    for doc in nlp.pipe(texts):
        # print(doc._.has_coref)
        # print(doc._.coref_clusters)
        # print(doc.text)
        # print(doc._.coref_resolved)
        for sent in doc.sents:
            if len(sent.text) > 10:
                sentences.append(sent.text)
    return sentences


wdparser_url = 'http://knowgen.sl.cloud9.ibm.com/wdparser'
mode = 'precision'  # mode = {'simple', 'precision', 'balanced', 'recall', 'thinker'}

ids, exp_docs = load_passage_sentences("/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json", 100)
out_fn = open ("/dccstor/myu/experiments/wikiparser/eli5_dev/eli5_wikiparse_dev_100.json", 'w')

for id, docs in zip(ids, exp_docs):
    sentences = docs2sents(docs)
    params = {'batch_input': json.dumps(sentences), 'mode': mode}
    response = requests.post(wdparser_url, data=params)
    output = response.json()
    output["id"] = id
    out_fn.write(json.dumps(output)+'\n')

