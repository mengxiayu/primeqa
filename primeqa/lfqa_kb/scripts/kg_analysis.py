import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
import pickle

stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS

Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])


def get_nonstop_words_pipe(doc):
    y = [
        token.lemma_ for token in doc if
        not token.is_stop
        and not token.is_currency
        and not token.is_digit
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and not token.pos_ == "PRON"
    ]
    return set(y)

def get_non_stop_words(passages):
    all_passages = []
    for doc in nlp.pipe(passages, n_process=4):
        words = get_nonstop_words_pipe(doc)
        all_passages.append(words)
    return all_passages

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def union(lst1, lst2):
    return list(set(lst1) & set(lst2))

def compute_overlaps(experiment, hops):
    gold_kg_avg_overlap = 0
    gen_kg_avg_overlap = 0
    gen_gold_avg_overlap = 0
    for index, example in hops.iterrows():
        experiment_paragraph = experiment[experiment["id"] == example.id].iloc[0]
        # gold_words = experiment_paragraph['max_rouge_ref'].lower().split()
        #generated_words = experiment_paragraph['prediction_text'].lower().split()
        
        gold_words, generated_words = get_non_stop_words([experiment_paragraph['max_rouge_ref'],experiment_paragraph['prediction_text']])

        if len(example['kg_vocab']) == 0:
            continue

        gold_intersection = intersection(gold_words,example['kg_vocab'])
        if len(gold_intersection) > 0:
            gold_kg_avg_overlap += len(gold_intersection)/len(example['kg_vocab'])
        generated_intersection = intersection(generated_words,example['kg_vocab'])
        if len(generated_intersection) > 0:
            gen_kg_avg_overlap += len(generated_intersection)/len(example['kg_vocab'])
        gen_gold = intersection(generated_words,gold_words)
        if len(gen_gold) > 0:
            gen_gold_avg_overlap += len(gen_gold)/len(gold_words)

        # if example['id'] in ids_to_watch:
        # # equal copy is higher than fid
        #     print(example['id'] + "\t gold \t" + 
        #     (str(gold_words) if gold_words != None else "") + "\n\t kg \t" +
        #     str(example['kg_vocab']) + "\n\t gen \t" + 
        #     (str(gen_gold) if  gen_gold != None else ""))
        #     print("gold/kg: " + str(gold_intersection) + " " + str(len(gold_intersection)/len(example['kg_vocab'])))
        #     print("gen/kg: " + str(generated_intersection) + " " + str(len(generated_intersection)/len(example['kg_vocab'])))
        #     print("gen/gold: " + str(gen_gold) + " " + str(len(gen_gold)/len(gold_words)))

    print("gold/kg: " + str(gold_kg_avg_overlap/len(hops)))
    print("gen/kg: " + str(gen_kg_avg_overlap/len(hops)))
    print("gen/gold: " + str(gen_gold_avg_overlap/len(hops)))

# equal_copy = pd.read_json('/dccstor/myu/experiments/eli5_cmsks_ctx_3/checkpoint-16452/output/eval_pred_scores.json', lines=True)
# fid = pd.read_json('/dccstor/myu/experiments/eli5_fid_beam_ctx3_0720/checkpoint-32904/output/eval_pred_scores.json', lines=True)
# kg_hop2 = pd.read_json('/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-kg-hop2.json', lines=True)
# kg_hop1 = pd.read_json('/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-kg-hop1.json', lines=True)
# kg_hop3 = pd.read_json('/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-kg-hop3.json', lines=True)

oracle = pd.read_json("/dccstor/srosent2/primeqa-mengxia/experiments/oracle_kg/train_kg_vocab/output/eval_pred_scores.json", lines=True)
fid = pd.read_json("/dccstor/srosent2/primeqa-mengxia/experiments/train_default/output/eval_pred_scores.json", lines=True)
#kg_hop2 = pd.read_json('/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-kg-hop2.json', lines=True)
oracle_kg = pd.read_json('/dccstor/srosent2/primeqa-mengxia/kg/dev/eli5-dev-kilt-oraclekg_best_ans_00.jsonl.gz', lines=True)

#all_answers = pickle.load(open("/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_ans_dev.pkl", 'rb'))
compute_overlaps(oracle, oracle_kg)
compute_overlaps(fid, oracle_kg)