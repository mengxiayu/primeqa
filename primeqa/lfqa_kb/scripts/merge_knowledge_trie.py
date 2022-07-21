import json
import os


def post_process_kg(n):
    path = "/dccstor/myu/experiments/knowledge_trie/eli5_openie_dev/"
    output_path = "/dccstor/myu/experiments/knowledge_trie/eli5_openie_dev_pkl/"
    assert os.path.exists(path)
    assert os.path.exists(output_path)

    # # create id_list
    # dataset_file = "/dccstor/myu/data/kilt_eli5/eli5-train-kilt.json"
    # id_list = []
    # with open(dataset_file, 'r') as f:
    #     for line in f:
    #         data = json.loads(line.strip())
    #         id_list.append(data["id"])
    
    # create a list of files
    # for train
    # kg_files = []
    # for i in range(54):
    #     start = i * 5000
    #     end = (i+1) * 5000
    #     kg_files.append(f"id2kg_{start}_{end}.json")
    # kg_files.append("id2kg_270000_272634.json")

    # for dev
    kg_files = ["id2kg_0_1507.json"]



    import marisa_trie  
    import pickle  
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser","tok2vec","ner"])
    cnt = 0
    x = kg_files[n]
    print(x)
    id2trie = {}
    with open(path + x) as f:
        for line in f:
            print(cnt)
            line_data = json.loads(line.strip())
            example_id = line_data["id"]
            # assert example_id == id_list[cnt]

            # TODO postprocess here
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

# import sys
# args = sys.argv
# post_process_kg(int(args[1]))
        
    
def merge_knowledge_tries():
    import pickle
    # create a list of files
    # for train
    train_path = "/dccstor/myu/experiments/knowledge_trie/eli5_openie_train_pkl/"
    kg_files = []
    for i in range(54):
        start = i * 5000
        end = (i+1) * 5000
        kg_files.append(f"{train_path}id2kg_{start}_{end}.pkl")
    kg_files.append(f"{train_path}id2kg_270000_272634.pkl")

    # for dev
    dev_path = "/dccstor/myu/experiments/knowledge_trie/eli5_openie_dev_pkl/"
    kg_files.append(f"{dev_path}id2kg_0_1507.pkl")

    id2trie_merge = {}
    for f in kg_files:
        id2trie = pickle.load(open(f, "rb"))
        for k,v in id2trie.items():
            id2trie_merge[k] = v
    print(len(id2trie_merge))
    with open("/dccstor/myu/experiments/knowledge_trie/eli5_openie_merge/id2kg.pickle", "wb") as f:
        pickle.dump(id2trie_merge, f)
    
merge_knowledge_tries()