# Given the following files build a new "passage" that is a oracle for the kg

# 1. OpenIE triples
# 2. Gold answers and DPR passage file
# Oracle: Use the question, answers and the triples to keep all triples that have words that are in the question and answer
# (and keep score to threshold later). 
# Filter out stopwords using NLTK to compute score and decide what kg triples to keep.
# Store everything in a new single file: new kg_vocab, kg_sentences, passages, answers

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import gzip
import sys
import string
from stanza.server import CoreNLPClient
import os
from tqdm import tqdm

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'
os.environ['CORENLP_HOME'] = '/afs/crc.nd.edu/user/m/myu2/stanford-corenlp-4.4.0'

from sentence_transformers import SentenceTransformer, util
import json
from nltk import sent_tokenize
import torch
sbert = SentenceTransformer('all-MiniLM-L6-v2')

def load_triples(input_file):
    with open (input_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
    return data_lines

def load_answers_and_passages(input_file):
    with gzip.open (input_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
    return data_lines

def get_overlap(question, answers, kg_triples, train=True):
    answer_words = []
    question_words = []
    # keep question as "answer" to include those words too
    for x in question.split():
        w = lemmatizer.lemmatize(x)
        if w not in stop_words and w not in string.punctuation:
            question_words.append(w.lower())

    answer_sents = []
    # add question to each answer
    for ans in answers:
        if train and ans['meta']['score'] < 3:
            continue
        words = set(question_words)
        if 'answer' in ans:
            for x in ans['answer'].split():
                w = lemmatizer.lemmatize(x)
                if w not in stop_words:
                    words.add(w.lower())
            answer_words.append(words)
            answer_sents.append(sent_tokenize(ans['answer']))

    # skip if there are no reliable answers (above threshold)
    if len(answer_words) == 0:
        return None, None, None

       
    kg_oracle_triples = []
    kg_oracle_sentences = {}
    kg_oracle_words = set()
    kg_all_words = set()
 
    for triple in kg_triples:
        kg_words = set()

        triple_parts = triple.split('\t')
        for part in triple_parts[:-1]: # object, relation, subject
            w = lemmatizer.lemmatize(part).lower()
            if w not in stop_words:
                kg_words.add(w)
        # counts = []
        # precisions = []
        for i, ans_words in enumerate(answer_words):
            oracle_words = kg_words & ans_words
            num_kg_words =  len(kg_all_words)
            kg_all_words.update(kg_words)
            kg_oracle_words.update(oracle_words)

            # only add to oracles if this triple has a new unseen word.
            new_words_added = False
            if len(kg_all_words) > num_kg_words:
                new_words_added = True

            cnt = len(oracle_words)
            p = cnt / len(kg_words) if cnt > 0 else 0
            if new_words_added and cnt > 0:
                kg_text = ' '.join(triple_parts[:-1])
                kg_embedding = sbert.encode(kg_text, convert_to_tensor=True)
                ans_sents = answer_sents[i]
                ans_embeddings = sbert.encode(ans_sents, convert_to_tensor=True)
                sbert_scores = util.cos_sim(kg_embedding, ans_embeddings)
                max_sbert_score = float(torch.max(sbert_scores))
                kg_oracle_triples.append({"text": ' '.join(triple_parts[:-1]), "count": cnt, "precision": p, "sbert_score": max_sbert_score})

                if triple_parts[-1] not in kg_oracle_sentences:
                    kg_oracle_sentences[triple_parts[-1]] = 0
                kg_oracle_sentences[triple_parts[-1]] += 1
                # counts.append(cnt)
                # precisions.append(p)
        # if len(counts) > 0:
        #     kg_oracle_triples.append({"triple":triple, "count": counts, "precision": precisions})
    if len(kg_oracle_triples) > 0: # sort by useful triple count
        kg_oracle_triples = sorted(kg_oracle_triples, key=lambda x: (x["count"], x["precision"]), reverse=True)
    if len(kg_oracle_sentences) > 0: # sort by useful triple count
        kg_oracle_sentences = sorted(kg_oracle_sentences.items(), key=lambda x: x[1], reverse=True)
    return kg_oracle_words, kg_oracle_triples, kg_oracle_sentences

# convert to vocab, sentence, and triple data
def get_sentences(passages, matched_ids, client, threshold=30):
    kg_sentences = []

    passages_by_id = {}

    for passage in passages:
        passages_by_id[passage['pid']] = passage

    count = 0
    for matched_id in matched_ids:
        count += 1
        passage_id, sentence_id = matched_id[0].split('-')
        passage = passages_by_id[passage_id]
        offsets = passage_id[passage_id.index("::")+2:]
        start = int(offsets.split(",")[0][1:])
        if start == 0:
            passage_text = passage["title"] + "." + passage['text'][len(passage["title"]):]
        else:
            passage_text = passage['text']
        doc = client.annotate(passage_text)
        sentence = ' '.join([token.word for token in doc.sentence[int(sentence_id)].token])
        kg_sentences.append({'sentence': sentence, 'count': matched_id[1]})
        if count > threshold:
            break

    return kg_sentences

def main():
    data_file = sys.argv[1] #"/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages/eli5-train-kilt-dpr-00.json.gz" 
    triple_file = sys.argv[2] #"/dccstor/srosent2/primeqa-mengxia/data/openie_triples_100passages_fixed/eli5-openie-triples-00.json"
    output_file = sys.argv[3] #"/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg/train/eli5-train-kilt-dpr-kg-00.json.gz"
    port = sys.argv[4]

    data = load_answers_and_passages(data_file)
    triples = load_triples(triple_file)

    # nlp = stanza.Pipeline(lang='en', processors='tokenize')
    with CoreNLPClient(annotators=['tokenize','ssplit','lemma'], #openie'], # tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref',
                   memory='16G', be_quiet=True, properties={}, endpoint="http://localhost:" + str(port)) as client:
        # process kg for oracle
        i = 0
        for line in tqdm(data,total=len(data)):
            assert(line['id'] == triples[i]['id'])
            kg_vocab, kg_triples, kg_sentences = get_overlap(line['input'], line['output'], triples[i]['kg'], train=False)

            # there are was no answer            
            if kg_vocab == None and kg_triples == None and kg_sentences == None:
                i += 1
                continue

            kg_sentences = get_sentences(line['passages'], kg_sentences, client, threshold=30)
            line['kg_vocab'] = list(kg_vocab)
            line['kg_triples'] = kg_triples
            line['kg_sentences'] = kg_sentences
            i += 1
    # write updated data with kg info to file
    with gzip.open (output_file, 'w') as f:
        for line in data:
            f.write((json.dumps(line) + "\n").encode('utf-8'))
if __name__ == '__main__':
    main()
