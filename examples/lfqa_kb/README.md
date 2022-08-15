# Long-form Question Answering

The code is implemented with Huggingface 4.17. The pipeline is modified from [this Transformers example](https://github.com/huggingface/transformers/blob/v4.17.0/examples/pytorch/question-answering/run_seq2seq_qa.py)

## Requirements
- primeqa
- datasets
- rouge
- spacy (Also download the model with this command `python -m spacy download en_core_web_lg`)
- marisa_trie

## Training
You can train the model with the following scripts.

### To Train BART
You can run the script with:
```sh
output_dir=/dccstor/myu/experiments/eli5_bart_beam_0719

jbsub -q x86_24h -cores 10+1 -mem 80g -require 'a100_80gb' -proj 'eli5' -name 'train_bart' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name eli5 \
  --train_file /dccstor/myu/data/kilt_eli5_dpr/eli5-train-kilt-dpr.json \
  --validation_file /dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 64 \
  --generation_max_length 256 \
  --max_answer_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --generation_num_beams 1 \
  --preprocessing_num_workers 10 \
```

### To Train BART + supporting passages
The `n_context` and the `context_column` should both be included in the script, in this case.
```sh
output_dir=/dccstor/myu/experiments/eli5_dprbart_large_beam_0719

jbsub -q x86_24h -cores 10+1 -mem 32g -require 'v100' -proj 'eli5' -name 'train_dprbart' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name eli5 \
  --train_file /dccstor/myu/data/kilt_eli5_dpr/eli5-train-kilt-dpr.json \
  --validation_file /dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --n_context 3 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --generation_max_length 256 \
  --max_answer_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --generation_num_beams 1 \
  --preprocessing_num_workers 10 \
  
```

### To Train FiD
```sh
output_dir=/dccstor/myu/experiments/eli5_fid_greedy_ctx3_3e05_0805

jbsub -q x86_24h -cores 10+1 -mem 40g -require 'a100_80gb' -proj 'eli5' -name 'train_fid' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_fid.py \
  --model_name_or_path facebook/bart-large \
  --train_file /dccstor/myu/data/kilt_eli5_dpr/eli5-train-kilt-dpr.json \
  --validation_file /dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size  32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --generation_max_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 2 \
  --n_context 3 \
  --preprocessing_num_workers 10 \
  --generation_num_beams 1 \
  # --max_train_samples 10000 \
```



### CopyScore / CopyEmbed
For CopyEmbed, change `run_cmsks_score.py` -> `run_cmsks_learn_embed.py`

```
output_dir=/dccstor/myu/experiments/eli5_cmsks_score_hop2_ctx3_greedy_3e05_0804
jbsub -q x86_24h -cores 10+1 -mem 40g -require 'a100_80gb' -proj 'eli5' -name 'cp_score_hop2_greedy' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_cmsks_score.py \
  --model_name_or_path facebook/bart-large \
  --train_file /dccstor/myu/data/kilt_eli5_dpr/eli5-train-kilt-dpr.json \
  --validation_file /dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --generation_max_length 256 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 2 \
  --n_context 3 \
  --kg_file /dccstor/myu/experiments/knowledge_trie/eli5_openie_merge/id2kg.pickle \
  --preprocessing_num_workers 10 \
  --generation_num_beams 1 \
```


### CopyScore / CopyEmbed + ANS_oracle
```
output_dir=/dccstor/myu/experiments/eli5_cmsks_score_oracle_ctx3_3e05_greedy_0803
jbsub -q x86_24h -cores 10+1 -mem 40g -require 'a100_80gb' -proj 'eli5' -name 'cp_score_orcl_grdy' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_cmsks_score_oracle.py \
  --model_name_or_path facebook/bart-large \
  --train_file /dccstor/myu/data/kilt_eli5_dpr/eli5-train-kilt-dpr.json \
  --validation_file /dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr-single.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --generation_max_length 256 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 3 \
  --n_context 3 \
  --kg_file /dccstor/myu/experiments/knowledge_trie/eli5_openie_merge/id2kg.pickle \
  --preprocessing_num_workers 10 \
  --generation_num_beams 1 \
```


### To FiD + KG (To be updated)
```
```

### DPR

*re2g_v2 branch*
```
export DS=eli5
export CORPUS=http://9.59.194.174:5001
working_dir=/dccstor/myu/retrieval
data_split=train

jbsub -q x86_24h -cores 1+1 -mem 32g -require 'v100' -proj 'oneqa' -name 'dpr_eli5_train' -o ${working_dir}/run_dpr_eli5_${data_split}.log \
python ${working_dir}/OneQA/examples/re2g/dpr/dpr_apply.py \
  --kilt_data ${working_dir}/data/KILT/${DS}/${DS}-${data_split}-kilt.json  \
  --output ${working_dir}/data/KILT/${DS}/predictions/dpr/eli5_re2g_${data_split}.json  --include_passages \
  --corpus_endpoint ${CORPUS} --n_docs_for_provenance 20 \
  --qry_encoder_path  /dccstor/few-shot-rel/eli5/models/re2g_nq/qry_encoder
```

### BART For ASQA dataset
```
output_dir=dccstor/myu/experiments/asqa_bart_large_beam
jbsub -q x86_24h -cores 1+1 -mem 32g -require 'v100' -proj 'asqa' -name 'train_dprbart_beam' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name asqa \
  --train_file /dccstor/myu/data/asqa/asqa_train.json \
  --validation_file /dccstor/myu/data/asqa/asqa_dev.json \
  --question_column ambiguous_question \
  --answer_column annotations \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 64 \
  --max_answer_length 128 \
  --generation_max_length 128 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 2 \
  --generation_num_beams 4
  # --dataset_name kilt_tasks \
  # --dataset_config_name eli5 \
```

### BART + supporting passages for ASQA
```
output_dir=/dccstor/myu/experiments/asqa_dprbart_beam_5e05
jbsub -q x86_24h -cores 4+1 -mem 32g -require 'v100' -proj 'asqa' -name 'train_asqa_dprbart' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name asqa \
  --train_file /dccstor/myu/data/asqa_dpr/asqa_train_dpr.json \
  --validation_file /dccstor/myu/data/asqa_dpr/asqa_dev_dpr.json \
  --question_column ambiguous_question \
  --answer_column annotations \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --max_seq_length 512 \
  --max_answer_length 128 \
  --generation_max_length 128 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 3 \
  --context_column passages \
  --n_context 5 \
  --generation_num_beams 4 \
  --preprocessing_num_workers 4 \
```