import pickle
import marisa_trie

def merge_knowledge_tries():
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