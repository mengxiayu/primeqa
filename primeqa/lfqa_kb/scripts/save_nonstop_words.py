import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])

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
        token.lemma_ for token in doc if
        not token.is_stop
        and not token.is_currency
        and not token.is_digit
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and not token.pos_ == "PROPN"
    ]
    return set(y)



def parse_train_passages():
    split = "train"
    data_file = f"/dccstor/myu/data/kilt_eli5_dpr/eli5-{split}-kilt-dpr.json"
    data_lines = []
    with open (data_file, 'r') as f:
        for line in f:
            data_lines.append(json.loads(line.strip()))
    all_passages = []
    all_passages_corpus = [' '.join([x["text"] for x in data["passages"]]) for data in data_lines]
    cnt = 0
    for doc in nlp.pipe(all_passages_corpus, n_process=4):
        words = get_nonstop_words_pipe(doc)
        all_passages.append(words)
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

    import pickle
    pickle.dump(all_passages, open(f"/dccstor/myu/experiments/eli5_analysis/all_psg_{split}.pkl", 'wb'))

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

    import pickle
    pickle.dump(all_passages, open(f"/dccstor/myu/experiments/eli5_analysis/all_psg_{split}_{topk}.pkl", 'wb'))

parse_dev_passages(topk=3)


