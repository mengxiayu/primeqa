# Given the answers and the retrieved passages, filter the answers to exclude any sentence
# that does not have any of the oracle words in it (or some threshold if this doesn't do much).
# STEP 1 load the files that have the kg info
# STEP 2 compare each answer to the KG vocab list
# STEP 3 filter answers to sentences that match KG

import os
import json
from stanza.server import CoreNLPClient

os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'

def load_answers_and_passages(input_file):
    with open (input_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
    return data_lines

def filter_gold_sentences(example, client):
    for answer in example['output']:
        kg_vocab = answer["kg_vocab"]
        answer_text = answer["answer"]

        doc = client.annotate(answer_text)

        sentences = []
        
        for sentence_i in doc.sentence:
            keep = False
            sentence_text = ""
            for token in sentence_i.token:
                sentence_text += token.word + " "
                if token.lemma.lower() in kg_vocab:
                    keep = True
                    kg_vocab.remove(token.lemma.lower())
            if keep:
                sentences.append(sentence_text)

        if len(sentences) == len(doc.sentence):
            continue
        if len(sentences) == 0:
            answer["filtered_answer"] = None
        else:
            answer["filtered_answer"] = ". ".join(sentences)
    return example



def main():
    data_file = "/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg_best_all_cased_pr/eli5-train-kilt-dpr-kg-00.json" 
    data_lines = load_answers_and_passages(data_file)
    port = 7500

    all_filtrered = 0

    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'lemma'], memory='16G', be_quiet=True, properties={}, endpoint="http://localhost:" + str(port)) as client:
        for example in  data_lines:
            filtered = 0
            non_filtered = 0
            example = filter_gold_sentences(example, client)
            for answer in example["output"]:
                if "filtered_answer" in answer:
                    filtered += 1
                else:
                    non_filtered += 1
            print("filtered: " + str(filtered) + ", non-filtered: " + str(non_filtered))
            if non_filtered == 0 and filtered > 0:
                all_filtered += 1
        print(all_filtered)

if __name__ == '__main__':
    main()
