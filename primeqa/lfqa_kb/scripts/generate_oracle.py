

# construct train/dev dataset for passage-answer overlap
# passage: top n passages
# answer: all answers that score >= 30
import json
import pickle
import glob
import gzip
import shutil
import spacy
import sys
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner","tok2vec"])
nlp.add_pipe('sentencizer')

split = sys.argv[1]
file_idx = sys.argv[2]

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

# for each word in the kg, find all the sentences that have those words.
def get_sentences(kg_words, passages, max_num_sentences=10):
    kg_sentences = {}
    sentence_count = {}

    for passage in passages:
        doc = nlp(passage['text'])
        
        # for each sentence, get the interesection of kg and sentence and keep track of all > 0
        for sentence in doc.sents:
            sentence_words = get_nonstop_words_pipe(sentence)
            overlap = list(sentence_words & set(kg_words))

            if len(overlap) > 0:
                sentence_count[sentence] = len(overlap)
                for word in overlap:
                    if word not in kg_sentences:
                        kg_sentences[word] = []
                    kg_sentences[word].append(sentence)
    seen = set()
    sorted_sentences = dict(sorted(sentence_count.items(), key=lambda item: item[1], reverse=True))

    final_sentences = []

    for sentence in sorted_sentences:
        if len(seen) == len(kg_sentences): # or len(final_sentences) >= max_num_sentences:
            break
        added = False
        for word in kg_sentences:
            if word not in seen and sentence in kg_sentences[word]:
                seen.add(word)
                if not added:
                    added = True
                    final_sentences.append({"text": str(sentence), "count": sorted_sentences[sentence]})                
    return final_sentences


# split = "train"
all_answers = []
all_passages = []

answer_files = glob.glob("/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_ans_" + split + "_" + file_idx + ".pkl")
answer_files.sort()
for file in answer_files:
    print(file)
    all_answers.extend(pickle.load(open(file, 'rb')))
passage_files = glob.glob("/dccstor/srosent2/primeqa-mengxia/kg/data_nostopwords/all_psg_" + split + "_" + file_idx + ".pkl")
passage_files.sort()
for file in passage_files:
    print(file)
    all_passages.extend(pickle.load(open(file, "rb")))

if split == "dev":
    data_file = "/dccstor/mabornea1/lfqa_kb_data/kilt_eli5_dpr/eli5-" + split + "-kilt-dpr.json"
else:
    data_file = "/dccstor/srosent2/primeqa-mengxia/data/eli5-train-kilt-dpr_" + file_idx + ".json.gz"

if data_file.endswith('gz'):
    with gzip.open (data_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
else:
    with open (data_file, 'r') as f:
        data_lines = [json.loads(line.strip()) for line in f]
print(len(data_lines))
assert len(all_answers) == len(all_passages) == len(data_lines)

output_dir = "/dccstor/srosent2/primeqa-mengxia/kg/" + split + "/"

# overlaps = []
fw_best = open(output_dir + "eli5-" + split + "-kilt-oraclekg_best_ans_" + file_idx + ".jsonl", 'w')
# fw_best_sentence = open(output_dir + "eli5-" + split + "-kilt-oraclekg_best_ans_sentence_" + file_idx + ".jsonl", 'wb')
# fw_all_sentence = open(output_dir + "eli5-" + split + "-kilt-oraclekg_all_ans_sentence_" + file_idx + ".jsonl", 'wb')
fw_all = open(output_dir + "eli5-" + split + "-kilt-oraclekg_all_ans_" + file_idx + ".jsonl", 'w')

avg_recall = 0
avg_best_count = 0
avg_all_count = 0
avg_best_answer_length = 0
avg_all_answer_length = 0
empty_kg = 0
count = 0

for idx, data in enumerate(data_lines):
    # answers_data = [x for x in data["output"]]
    if split == 'dev':
        answers_data = [x for x in data["output"] if "answer" in x]
    else:
        answers_data = [x for x in data["output"] if "answer" in x and x["meta"]["score"] >= 3]
    kg_data = []
    assert len(answers_data) == len(all_answers[idx])

    words_psg = all_passages[idx]
    
    max_recall = 0.0
    best_answer = None
    best_overlap = [0,0,0,0]
    all_answer_words = set()
    avg_answer_length = 0

    for i, words_ans in enumerate(all_answers[idx]):
        word_overlap = words_psg & words_ans
        all_answer_words = all_answer_words.union(words_ans)
        avg_answer_length += len(words_ans)

        if len(words_ans) == 0:
            print(answers_data[i]["answer"])
        recall = len(word_overlap)/len(words_ans) if len(words_ans) > 0 else 0
        if split == "dev":
                answers_data[i]["meta"] = {}
        answers_data[i]["meta"]["recall"] = recall
        if recall >= max_recall:
            max_recall = recall
            best_answer = answers_data[i]
            kg_data = list(word_overlap)
            best_overlap = [len(word_overlap), recall, len(word_overlap)/len(words_psg), len(words_ans)]
    if len(all_answers[idx]) > 0:
        avg_all_answer_length += avg_answer_length/len(all_answers[idx])
    avg_best_answer_length += best_overlap[3]
    avg_recall += max_recall
    avg_best_count += len(kg_data)
    if len(kg_data) == 1:
        empty_kg += 1
    
    avg_all_count += len(list(words_psg & all_answer_words))
    data["output"] = answers_data
    data["kg_vocab"] = kg_data
    data["kg_sentences"] = get_sentences(data["kg_vocab"], data["passages"])
    fw_best.write((json.dumps(data)+ '\n'))  
    data["kg_vocab"] = list(words_psg & all_answer_words)
    data["kg_sentences"] = get_sentences(data["kg_vocab"], data["passages"])
    fw_all.write((json.dumps(data) + '\n'))
    count += 1

    if count % 100 == 0:
        print(count)

fw_best.close()
fw_all.close()

print("avg best kg recall")
print(avg_recall/len(data_lines))
print("avg best kg count")
print(avg_best_count/len(data_lines))
print("avg all kg count")
print(avg_all_count/len(data_lines))
print("avg best answer length")
print(avg_best_answer_length/len(data_lines))
print("avg all answer length")
print(avg_all_answer_length/len(data_lines))
print("kg len == 1")
print(empty_kg)

print("# examples in " + split)
print(len(data_lines))

# import os
# def compress_file(file_name):
#     fw_in = open(file_name, 'rb')

#     with gzip.open(file_name + ".gz", 'wb') as f_out:
#         shutil.copyfileobj(fw_in, f_out)
#     os.remove(file_name)

# compress_file(output_dir + "eli5-" + split + "-kilt-oraclekg_best_ans_" + file_idx + ".jsonl")
# # compress_file(output_dir + "eli5-" + split + "-kilt-oraclekg_best_ans_sentence_" + file_idx + ".jsonl")
# # compress_file(output_dir + "eli5-" + split + "-kilt-oraclekg_all_ans_sentence_" + file_idx + ".jsonl")
# compress_file(output_dir + "eli5-" + split + "-kilt-oraclekg_all_ans_" + file_idx + ".jsonl")
