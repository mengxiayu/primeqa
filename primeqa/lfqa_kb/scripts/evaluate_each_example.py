from rouge import Rouge
import numpy as np
import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter, force=True)
nlp = spacy.load("en_core_web_lg", disable=["parser","ner"])


def get_nonstop_words(x):
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
    return set(y)

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

pred_fn = "/dccstor/myu/experiments/eli5_bart_beam_0719/eval_predictions.json"
data_fn = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json"
output_fn = "/dccstor/myu/experiments/eli5_bart_beam_0719/eval_pred_scores.json"
print(pred_fn)
print(output_fn)

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
        pred = predictions[idx]['prediction_text']
        ref = max_score_refs[idx]
        words_pred = get_nonstop_words(pred)
        words_ref = get_nonstop_words(ref)
        tmp_record = {
            "id" : predictions[idx]['id'],
            "input": questions[idx],
            "prediction_text" : pred,
            "max_rouge_ref" : ref,
            "rouge-L": f"{line*100:.2f}",
            "overlaps": {"count": len(words_pred & words_ref), "precision": (len(words_pred & words_ref)/len(words_pred))*100, "recall": (len(words_pred & words_ref)/len(words_ref))*100}
        }
        f.write(json.dumps(tmp_record) + '\n')
    print("saved generation scores.")