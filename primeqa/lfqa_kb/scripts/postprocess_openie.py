import json
import os
import marisa_trie  
import pickle  
import spacy
import sys

def post_process_kg(split, n, path, output_path):
    '''
    convert strings to marisa_trie
    '''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    assert os.path.exists(path)
    assert os.path.exists(output_path)

    # create a list of files
    if split == "train":
        # dataset_file = "/dccstor/myu/data/kilt_eli5/eli5-train-kilt.json"
        # with open(dataset_file, 'r') as f:
        #     id_list = [json.loads(line.strip())["id"] for line in f]
        kg_files = []
        for i in range(54):
            start = i * 5000
            end = (i+1) * 5000
            kg_files.append(f"id2kg_{start}_{end}.json")
        kg_files.append("id2kg_270000_272634.json")

    # for dev
    elif split == "dev":
        kg_files = ["id2kg_0_1507.json"]

    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    cnt = 0
    x = kg_files[n]
    # print(x)
    id2trie = {}
    with open(path + x) as f:
        for line in f:
            print(cnt)
            line_data = json.loads(line.strip())
            example_id = line_data["id"]
            # assert example_id == id_list[cnt]

            str_list = line_data["kg"]
            new_str_list = []
            for s in str_list:
                subj, obj = s.split('\t')
                subj_tokens = [t.lemma_ for t in nlp(subj) if not t.is_stop]
                if len(subj_tokens) == 0:
                    continue
                obj_tokens = [t.lemma_ for t in nlp(obj) if not t.is_stop]
                if len(obj_tokens) == 0:
                    continue
                new_s = " ".join(subj_tokens + obj_tokens)
                new_str_list.append(new_s)

            trie = marisa_trie.Trie(new_str_list)
            id2trie[example_id] = trie
            cnt += 1
    pickle.dump(id2trie, open(output_path + x.split('.')[0]+".pkl", 'wb'))
    print(cnt)


args = sys.argv
post_process_kg(args[1], int(args[2]), args[3], args[4])
        
    
