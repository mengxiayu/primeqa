import sys
sys.path.append("/dccstor/srosent2/primeqa-mengxia/primeqa/primeqa/")
# print(sys.path)
from lfqa_kb.metrics.utils import _metric_max_over_ground_truths
import pandas as pd

def rouge(ref_data, prediction_file):
    baseline_df = pd.read_json(prediction_file, lines=True, dtype=str)  #orient='records', lines=True
    baseline_df['hf_rouge'] = 0.0
    baseline_df['kilt_rouge'] = 0.0

    for i, row in baseline_df.iterrows():
        answers = []
        output = ref_data[row["id"]]['output']

        for o in output:
            if "answer" in o:
                answers.append(o["answer"])

        hf_rouge_score, kilt_rouge_score = _metric_max_over_ground_truths(row["text"],answers) # prediction_text
        
        baseline_df.loc[i, 'hf_rouge'] = hf_rouge_score
        baseline_df.loc[i, 'kilt_rouge'] = kilt_rouge_score
        
    hf_score = baseline_df['hf_rouge'].mean()
    kilt_score = baseline_df['kilt_rouge'].mean()
    return hf_score, kilt_score

# Gold Data

import json
ELI5 = {}

with open("/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg_best_all_cased_pr/eli5-dev-kilt-dpr-kg-00.json",'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        ELI5[data['id']] = data


import glob
pred_files = glob.glob("/dccstor/srosent2/generative/baseline_llms/ELI5-full/512-limit/closed-book/flan-t5-large/prefix_default-passages_False-0shot_pktemp-1.0_5_0.7-minmaxtok_100_1024/predictions-0-1507.json")

for file in pred_files:
    hf_score, kilt_score = rouge(ELI5, file)
    print(f"{file}|{hf_score}|{kilt_score}", flush=True)