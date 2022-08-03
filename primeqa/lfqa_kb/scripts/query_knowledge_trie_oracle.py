import pickle
import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
import marisa_trie
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)

nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])

def get_knowledge_vocab(ext_trie, local_kg, max_hops):
        # str_list = self.knowledge_trie[exp_id] # TODO pass it from the dataset
        # ext_trie = marisa_trie.Trie(str_list)
    tmp_kg = local_kg
    related_kgs = set()
    for i in range(max_hops):
        new_knowledge = []
        for ent in tmp_kg:
            for span in ext_trie.keys(ent):
                new_knowledge.extend(span.split(' '))
        new_knowledge = set(new_knowledge)
        tmp_kg = list(new_knowledge)
        related_kgs |= new_knowledge # this is the kg vocab
    return list(related_kgs)

def get_initial_query(x):
    y = [
        token.lemma_ for token in nlp(x) if
        not token.is_stop
        and not token.is_currency
        and not token.is_digit
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and not token.pos_ == "PROPN"
    ]
    return list(set(y))

kg_file = "/dccstor/myu/experiments/knowledge_trie/eli5_openie_merge/id2kg.pickle"
data_file = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-single.json"
output_file = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-kg-oracle.json"
max_hops = 1

# knowledge_tries = pickle.load(open(kg_file, 'rb'))


with open(data_file, 'r') as f:
    data_lines = []
    for line in f:
        data = json.loads(line.strip())
        answer = data["output"][0]["answer"]
        # example_id = data["id"]
        # ext_trie = knowledge_tries[example_id]
        query = get_initial_query(answer)

        kg_vocab = query
        print(kg_vocab)
        # data["query"] = query
        data["kg_vocab"] = kg_vocab
        data.pop("passages", None)
        data_lines.append(json.dumps(data))
print(len(data_lines))
with open (output_file, 'w') as f:
    for line in data_lines:
        f.write(line + '\n')
print("output file saved")
        
        

