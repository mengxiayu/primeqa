
fn_prediction = "/dccstor/myu/experiments/eli5_fid_beam_ctx3_0720/checkpoint-32904/output/eval_pred_scores.json"
fn_data = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json"
fn_out = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-single.json"

import json
with open(fn_prediction) as f:
    preds = [json.loads(line.strip()) for line in f]
with open(fn_data) as f:
    data = [json.loads(line.strip()) for line in f]

with open(fn_out, 'w') as f:
    for _pred, _data in zip(preds, data):
        assert _pred["id"] == _data["id"]
        ex = _data
        ex["output"] = [{"answer": _pred["max_rouge_ref"]}]
        f.write(json.dumps(ex) + '\n')
