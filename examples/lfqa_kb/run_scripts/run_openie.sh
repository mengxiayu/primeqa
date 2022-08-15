start=0
end=10
data_file="/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json"
output_dir="/dccstor/myu/data/openie/knowledge_trie/eli5_openie_dev/"
port='http://localhost:9090'

jbsub -q x86_24h -cores 1 -mem 8g  -proj 'eli5' -name 'openie_dev' -o /dccstor/myu/experiments/run_openie_eli5_${start}_${end}.log \
python /dccstor/myu/primeqa/primeqa/lfqa_kb/scripts/run_openie.py ${start} ${end} ${data_file} ${output_dir} ${port}\

# python /dccstor/myu/primeqa/primeqa/lfqa_kb/scripts/run_openie.py ${start} ${end} ${data_file} ${output_dir} ${port}