pred_file="/dccstor/myu/experiments/eli5_fid_beam_ctx3_0720/checkpoint-32904/output/eval_predictions.json"
pred_reformat_file="/dccstor/myu/experiments/eli5_fid_beam_ctx3_0720/checkpoint-32904/output/eval_predictions_reformat.json"
kilt_file="/dccstor/myu/data/kilt_eli5/eli5-dev-kilt.json"
echo ${pred_file}
python /dccstor/myu/primeqa/primeqa/lfqa_kb/scripts/reformat_prediction_for_kilt.py --pred_file ${pred_file} --kilt_file ${kilt_file}
python /dccstor/myu/KILT/kilt/eval_downstream.py --guess ${pred_reformat_file}  --gold ${kilt_file}