from rouge import Rouge
from transformers.trainer_utils import EvalPrediction
import numpy as np
from rouge_score import rouge_scorer
import sys

hf_rouge = rouge_scorer.RougeScorer(rouge_types=['rougeLsum'], split_summaries=True) #evaluate.load('rouge')
kilt_rouge = Rouge(metrics=['rouge-l'])
sys.setrecursionlimit(20000)

def _rougel_score(prediction, ground_truth):
    try:
        hf_scores = hf_rouge.score(ground_truth, prediction)
        kilt_scores = kilt_rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0, 0.0
    return hf_scores['rougeLsum'].fmeasure, kilt_scores["rouge-l"]["f"]

def _metric_max_over_ground_truths(prediction, ground_truths):
    hf_scores_for_ground_truths = []
    kilt_scores_for_ground_truths = []
    for ground_truth in ground_truths:
        hf_score, kilt_score = _rougel_score(prediction, ground_truth)
        hf_scores_for_ground_truths.append(hf_score)
        kilt_scores_for_ground_truths.append(kilt_score)
    max_hf = 0 if len(hf_scores_for_ground_truths) == 0 else max(hf_scores_for_ground_truths)
    max_kilt = 0 if len(kilt_scores_for_ground_truths) == 0 else max(kilt_scores_for_ground_truths)      
    return max_hf, max_kilt

# modified from 
# https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/examples/pytorch/summarization/run_summarization.py#L563
def compute_metrics(p: EvalPrediction):
    # adopted KILT standard evaluation from https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
    total_count = 0
    hf_rougel = 0
    kilt_rougel = 0
    preds =p.predictions
    refs = p.label_ids

    for pred,ref in zip(preds,refs):
        _id = pred["id"]
        _pred = pred["prediction_text"]
        assert ref["id"] == _id
        total_count += 1
        _refs = ref["answers"]
        local_hf_rougel, local_kilt_rougel = _metric_max_over_ground_truths(_pred, _refs)
        pred['hf_rougeL'] = local_hf_rougel
        pred['kilt_rougeL'] = local_kilt_rougel
        hf_rougel += local_hf_rougel
        kilt_rougel += local_kilt_rougel
    # result = metric.compute(predictions=preds, references=refs)
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {"hf_rougeL": (hf_rougel/total_count)*100, "kilt_rougeL": (kilt_rougel/total_count)*100}
    prediction_lens = [pred["prediction_text"].count(' ') for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

