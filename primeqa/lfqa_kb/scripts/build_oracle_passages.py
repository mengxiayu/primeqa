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
from rouge import Rouge

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'
rouge = Rouge()

def load_triples(input_file):
    with open (input_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
    return data_lines

def load_answers_and_passages(input_file):
    with gzip.open (input_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
    return data_lines

def get_best_answer(answers):
    best = None
    max_rouge = 0

    for answer in answers:
        if 'answer' not in answer or answer['meta']['score'] < 3:
            continue        
        if answer['meta']['rouge'] > max_rouge:
            max_rouge = answer['meta']['rouge']
            best = answer
    return best

def compute_rouge_answer(answers):

    # answers_to_check = []

    # for answer in answers:
    #     if 'answer' not in answer:
    #         continue
    #     if answer['meta']['score'] < 3:
    #         continue
    #     answers_to_check.append(answer)

    # answers = answers_to_check

    # if len(answers) == 0:
    #     return []

    # scores = [0] * len(answers)
    for i in range(len(answers)):
        if 'answer' not in answers[i]:
                continue

        for j in range(i+1, len(answers)):
            if 'answer' not in answers[j]:
                continue
            if 'meta' not in answers[i]:
                answers[i]['meta'] = {}
            if 'rouge' not in answers[i]['meta']:
                    answers[i]['meta']['rouge'] = 0
            try:
                score = rouge.get_scores(answers[i]['answer'], answers[j]['answer'], avg=True)
  
                if 'meta' not in answers[j]:
                    answers[j]['meta'] = {}
                if 'rouge' not in answers[j]['meta']:
                    answers[j]['meta']['rouge'] = 0    
                answers[i]['meta']['rouge'] += score["rouge-l"]["f"]
                answers[j]['meta']['rouge'] += score["rouge-l"]["f"]
            except ValueError:  # "Hypothesis is empty."
                return 0.0
    return answers

def get_answer_words(question, answer, client, train=True):
    question_words = []

    # if len(answers) == 0:
    #     return answer_words

    # keep question as "answer" to include those words too
    doc = client.annotate(question)
    for sentence in doc.sentence:
        for token in sentence.token:
            if token.lemma.lower() not in stop_words and token.lemma.lower() not in string.punctuation:
                question_words.append(token.lemma)


    if train and answer['meta']['score'] < 3:
        return None
    words = set(question_words)
    if 'answer' in answer:
        doc = client.annotate(answer['answer'])
        for sentence in doc.sentence:
            for token in sentence.token:
                if token.lemma.lower() not in stop_words and token.lemma.lower() not in string.punctuation:
                    words.add(token.lemma)
    return words

def get_overlap(line, kg_triples, client, train=True, best=True):
    answer_words = []
    
    question = line['input']
    answers = line['output']
    line['kg_vocab'] = set()
    line['kg_sentences'] = set()
    line['kg_triples'] = set()

    answers = compute_rouge_answer(answers)

    if best and train:
        answers = get_best_answer(answers)

    passages_by_id = {}

    for passage in line["passages"]:
        passages_by_id[passage['pid']] = {"passage": passage}

    kg_oracle_sentences_all = {}
    kg_oracle_triples_all = []
    kg_all_words_all = set()
    kg_oracle_words_all = set()

    for answer in answers:
    
        answer_words = get_answer_words(question, answer, client, train)
        
        # skip if there are no reliable answers (above threshold)
        if len(answer_words) == 0:
            continue
        
        kg_oracle_triples = []
        kg_oracle_sentences = {}
        kg_oracle_words = set()
        kg_all_words = set()
    
        for triple in kg_triples:
            kg_words = set()

            triple_parts = triple.split('\t')
            for part in triple_parts[:-1]: # object, relation, subject
                words = part.split()
                for word in words:
                    w = lemmatizer.lemmatize(word)
                    if w.lower() not in stop_words:
                        kg_words.add(w)

            
            oracle_words = kg_words & answer_words
            num_kg_words =  len(kg_all_words)
            num_kg_words_all =  len(kg_all_words_all)
            kg_all_words.update(kg_words)
            kg_oracle_words.update(oracle_words)
            kg_all_words_all.update(kg_words)
            kg_oracle_words_all.update(oracle_words)

            # only add to oracles if this triple has a new unseen word.
            new_words_added = False
            if len(kg_all_words) > num_kg_words:
                new_words_added = True
            new_words_added_all = False
            if len(kg_all_words_all) > num_kg_words_all:
                new_words_added_all = True

            cnt = len(oracle_words)
            p = cnt / len(kg_words) if cnt > 0 else 0
            if new_words_added and cnt > 0:
                # kg_text = ' '.join(triple_parts[:-1])
                # kg_embedding = sbert.encode(kg_text, convert_to_tensor=True)
                # ans_sents = answer_sents[i]
                # ans_embeddings = sbert.encode(ans_sents, convert_to_tensor=True)
                # sbert_scores = util.cos_sim(kg_embedding, ans_embeddings)
                # max_sbert_score = float(torch.max(sbert_scores))
                kg_oracle_triples.append({"text": ' '.join(triple_parts[:-1]), "count": cnt, "precision": p}) #, "sbert_score": max_sbert_score})

                if triple_parts[-1] not in kg_oracle_sentences:
                    kg_oracle_sentences[triple_parts[-1]] = 0
                kg_oracle_sentences[triple_parts[-1]] += 1
            if new_words_added_all and cnt > 0:
                kg_oracle_triples_all.append({"text": ' '.join(triple_parts[:-1]), "count": cnt, "precision": p})

                if triple_parts[-1] not in kg_oracle_sentences_all:
                    kg_oracle_sentences_all[triple_parts[-1]] = 0
                kg_oracle_sentences_all[triple_parts[-1]] += 1
        if len(kg_oracle_triples) > 0: # sort by useful triple count
            kg_oracle_triples = sorted(kg_oracle_triples, key=lambda x: (x["count"], x["precision"]), reverse=True)
        if len(kg_oracle_sentences) > 0: # sort by useful triple count
            kg_oracle_sentences = sorted(kg_oracle_sentences.items(), key=lambda x: x[1], reverse=True)
        answer['kg_vocab'] = list(kg_oracle_words)
        answer['kg_sentences'] = get_sentences(passages_by_id, list(kg_oracle_sentences), client, threshold=30)
        answer['kg_triples'] = list(kg_oracle_triples)
    if len(kg_oracle_triples_all) > 0: # sort by useful triple count
            kg_oracle_triples_all = sorted(kg_oracle_triples_all, key=lambda x: (x["count"], x["precision"]), reverse=True)
    if len(kg_oracle_sentences_all) > 0: # sort by useful triple count
            kg_oracle_sentences_all = sorted(kg_oracle_sentences_all.items(), key=lambda x: x[1], reverse=True)
    line['kg_vocab'] = list(kg_oracle_words_all)
    line['kg_triples'] = list(kg_oracle_triples_all)
    line['kg_sentences'] = get_sentences(passages_by_id, list(kg_oracle_sentences_all), client, threshold=30)
    return line

# convert to vocab, sentence, and triple data
def get_sentences(passages_by_id, matched_ids, client, threshold=30):
    kg_sentences = []

    count = 0
    for matched_id in matched_ids:
        count += 1
        if count > threshold:
            break

        passage_id, sentence_id = matched_id[0].split('-')

        passage = passages_by_id[passage_id]["passage"]
        if "sentences" in passage:
            kg_sentences.append({'sentence': passage["sentences"][int(sentence_id)], 'count': matched_id[1]})
            continue
    
        offsets = passage_id[passage_id.index("::")+2:]
        start = int(offsets.split(",")[0][1:])
        if start == 0:
            passage_text = passage["title"] + "." + passage['text'][len(passage["title"]):]
        else:
            passage_text = passage['text']
        doc = client.annotate(passage_text)
        all_sentences = []
        for sentence_i in doc.sentence:
            sentence = ' '.join([token.word for token in sentence_i.token])
            all_sentences.append(sentence)
        kg_sentences.append({'sentence': all_sentences[int(sentence_id)], 'count': matched_id[1]})
        passage['sentences'] = all_sentences


    return kg_sentences

def main():
    data_file = sys.argv[1] #"/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages/eli5-train-kilt-dpr-00.json.gz" 
    triple_file = sys.argv[2] #"/dccstor/srosent2/primeqa-mengxia/data/openie_triples_100passages_fixed/eli5-openie-triples-00.json"
    output_file = sys.argv[3] #"/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_with_bestkg/train/eli5-train-kilt-dpr-kg-00.json.gz"
    port = sys.argv[4]

    data = load_answers_and_passages(data_file)
    triples = load_triples(triple_file)

    # nlp = stanza.Pipeline(lang='en', processors='tokenize')
    with CoreNLPClient(annotators=['tokenize','ssplit','lemma'], #openie'], # tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref',
                   memory='16G', be_quiet=True, properties={}, endpoint="http://localhost:" + str(port)) as client:
        # process kg for oracle
        do_train = False
        i = 0
        for line in tqdm(data,total=len(data)):
            assert(line['id'] == triples[i]['id'])

            if do_train == False:
                while 'answer' not in line['output'][-1]:
                    line['output'].pop()

            line = get_overlap(line, triples[i]['kg'], client, train=do_train)

            # # there are was no answer            
            # if kg_vocab == [] and kg_triples == [] and kg_sentences == []:
            #     line['kg_vocab'] = kg_vocab
            #     line['kg_triples'] = kg_triples
            #     line['kg_sentences'] = kg_sentences
            # else:
            #     kg_sentences = get_sentences(line['passages'], kg_sentences, client, threshold=30)
            #     line['kg_vocab'] = list(kg_vocab)
            #     line['kg_triples'] = kg_triples
            #     line['kg_sentences'] = kg_sentences
            i += 1
    # write updated data with kg info to file
    with gzip.open (output_file, 'w') as f:
        for line in data:
            f.write((json.dumps(line) + "\n").encode('utf-8'))
if __name__ == '__main__':
    main()
