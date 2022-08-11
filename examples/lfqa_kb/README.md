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

### To Train DPR+BART
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



### CopyScore
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

### CopyEmbed
```
output_dir=/dccstor/myu/experiments/eli5_cmsks_embed_hop2_ctx3_greedy_3e05_0804
jbsub -q x86_24h -cores 10+1 -mem 40g -require 'a100_80gb' -proj 'eli5' -name 'cp_embed_hop2_greedy' -o ${output_dir}/train.log \
python /dccstor/myu/primeqa/examples/lfqa_kb/run_cmsks_learn_embed.py \
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

### To FiD + KG (To be updated)
```
```

### DPR

*re2g_v2 branch*
