import json
import spacy
import sys
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner","tok2vec"])
import pickle

# def get_nonstop_words(x):
#     y = [
#         token.lemma_ for token in nlp(x) if
#         not token.is_stop
#         and not token.is_currency
#         and not token.is_digit
#         and not token.is_punct
#         and not token.is_space
#         and not token.like_num
#         and not token.pos_ == "PROPN"
#     ]
#     return set(y)


def get_nonstop_words_pipe(doc):
    y = [
        token.lemma_.lower() for token in doc if
        not token.is_stop
        and not token.is_currency
        and not token.is_digit
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and not token.pos_ == "PRON"
    ]
    return set(y)

def stopword_pipeline(corpus, status=False, n_process=4):
    data = []
    cnt = 0
    for doc in nlp.pipe(corpus, n_process=n_process):
        words = get_nonstop_words_pipe(doc)
        data.append(words)
        if status:
            cnt += 1
            if cnt % 100 == 0:
                print(cnt)
    return data

def parse_passages(data_file, split, idx):
    # split = "train"
    # data_file = f"/dccstor/mabornea1/lfqa_kb_data/kilt_eli5_dpr/eli5-{split}-kilt-dpr.json"
    data_lines = []
    with open (data_file, 'r') as f:
        for line in f:
            data_lines.append(json.loads(line.strip()))
    
    all_passages_corpus = []
    all_answers = []
    cnt = 0
    for data in data_lines:
        cnt += 1
        passage_text = ""
        for x in data["passages"]:
            passage_text += x["text"] + " "
        answer_text = []
        for x in data["output"]:
            if "answer" in x:
                if split == "train" and x["meta"]["score"] < 3:
                    continue
                answer_text.append(x["answer"])
        all_passages_corpus.append(passage_text[:-1])
        all_answers.append(stopword_pipeline(answer_text, n_process=1))

        if cnt % 100 == 0:
            print(cnt)
    
    all_passages = stopword_pipeline(all_passages_corpus, status=True)
    # all_answers = stopword_pipeline(all_answers_corpus, True)
    
    print("write " + str(len(all_answers)) + "to /dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_psg_" + split + "_" + idx + ".pkl")
    with open("/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_psg_" + split + "_" + idx + ".pkl", 'wb') as handle:
        pickle.dump(all_passages, handle)

    print("write " + str(len(all_answers)) + "lines to /dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_ans_" + split + "_" + idx + ".pkl")
    with open("/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_ans_" + split + "_" + idx + ".pkl", 'wb') as handle:
        pickle.dump(all_answers, handle)

def parse_dev_passages(topk):
    
    split = "dev"
    data_file = f"/dccstor/myu/data/kilt_eli5_dpr/eli5-{split}-kilt-dpr.json"
    data_lines = []
    with open (data_file, 'r') as f:
        for line in f:
            data_lines.append(json.loads(line.strip()))
    all_passages = []
    all_passages_corpus = [' '.join([x["text"] for x in data["passages"]][:topk]) for data in data_lines]
    cnt = 0
    for doc in nlp.pipe(all_passages_corpus, n_process=4):
        words = get_nonstop_words_pipe(doc)
        all_passages.append(words)
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

    pickle.dump(all_passages, open(f"/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_psg_{split}_{topk}.pkl", 'wb'))

# parse_dev_passages(topk=3)
# parse_passages("dev")

data_file = sys.argv[1]
split = sys.argv[2]
idx = sys.argv[3]

parse_passages(data_file, split, idx)