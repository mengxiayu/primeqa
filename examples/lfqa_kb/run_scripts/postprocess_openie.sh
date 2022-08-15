n=0
split=dev
path=/dccstor/myu/experiments/knowledge_trie/eli5_openie_${split}/
output_path=/dccstor/myu/experiments/knowledge_trie/eli5_openie_${split}_pkl/
jbsub -q x86_24h -cores 1 -mem 8g  -proj 'eli5' -name 'openie' -o /dccstor/myu/experiments/knowledge_trie/postprocess_${n}.log \
python /dccstor/myu/primeqa/primeqa/lfqa_kb/scripts/postprocess_openie.py ${split} ${n} ${path} ${output_path} \