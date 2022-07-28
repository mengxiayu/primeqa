from rouge import Rouge
import numpy as np
import json

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    max_idx = np.argmax(scores_for_ground_truths)
    return scores_for_ground_truths[max_idx], ground_truths[max_idx] 

pred_fn = "/dccstor/mabornea2/eli5/code/lfqa-primeqa/experiments/eli5_cmsks_learn_ctx3_hop2_learn_bs1_1e-5_0726/eval_predictions.json"
data_fn = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json"
output_fn = "/dccstor/mabornea2/eli5/code/lfqa-primeqa/experiments/eli5_cmsks_learn_ctx3_hop2_learn_bs1_1e-5_0726/eval_pred_scores.json"


with open(data_fn, 'r', encoding='utf-8') as f:
    references = []
    questions = []
    for line in f:
        data = json.loads(line.strip())
        _refs = [x["answer"] for x in data["output"] if "answer" in x]
        _q = data["input"]
        references.append(_refs)
        questions.append(_q)

with open(pred_fn, 'r') as f:
    predictions = json.loads(f.read())



rouge_scores = []
max_score_refs = []
for idx,line in enumerate(predictions):

    _refs = references[idx]
    _pred = line["prediction_text"]

    local_rougel, max_score_ref = _metric_max_over_ground_truths(
        _rougel_score, _pred, _refs
    )
    rouge_scores.append(local_rougel)
    max_score_refs.append(max_score_ref)

with open(output_fn, 'w') as f:
    for idx,line in enumerate(rouge_scores):
        tmp_record = {
            "id" : predictions[idx]['id'],
            "input": questions[idx],
            "prediction_text" : predictions[idx]['prediction_text'],
            "max_rouge_ref" : max_score_refs[idx],
            "rouge-L": f"{line*100:.2f}",
        }
        f.write(json.dumps(tmp_record) + '\n')
    print("saved generation scores.")