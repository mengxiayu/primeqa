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