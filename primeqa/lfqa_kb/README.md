# Long-form Question Answering
## Metric
`metrics.utils` includes Rouge-L scores adopted from KILT

## Models
- `cmsks_embed_oracle`: KG embeddings + learned p as weight +  ANS_oracle
- `cmsks_embed`: KG embeddings + learned p as weight + n-hop KG
- `cmsks_learnproj_norm`: copy in probability space (after softmax) + learned p as weight + KG-probability projection + n-hop KG
- `cmsks_learnproj_oracle`: copy in logit space + learned p as weight + KG-logit projection + ANS_oracle
- `cmsks_learnproj`: copy in logit space + learned p as weight + KG-logit projection + n-hop KG
- `cmsks_learnprojminmax_oracle`: minmax KG-logit intialization [max(lm_logits) for kg words, min(lm_logits) for non-kg words] + copy in logit space + learned p as weight + KG-logit projection + ANS_oracle
- `cmsks_learnprojminmax`: minmax KG-logit intialization + copy in logit space + learned p as weight + KG-logit projection + n-hop KG
- `cmsks_minmax`: minmax KG-logit intialization + copy in logit space + learned p as weight + n-hop KG
- `cmsks_score_oracle`: learned p as KG logits + ANS_oracle
- `cmsks_score`: learned p as KG logits + n-hop KG
- `cmsks_static_equal`: equal copy in logit space + n-hop KG
- `cmsks_static_equal_oracle`: equal copy in logit space + ANS_oracle
- `cmsks`: copy in logit spce + learned p as weight + n-hop KG
- `fid`: the original FiD model

## Processor
`processors.postprocessors` gets multiple references for each question.


`processors.preprocessors` includes various implementation for different input setting. For each setting, there are always three functions, one for batching, one for train set preprocess, one for dev set preprocess.

In `eli5.py` we have:
- For BART: `preprocess_eli5_validation_function`, `preprocess_eli5_function`, `preprocess_eli5_batch`

- For FiD: `encode_passages`, `preprocess_eli5_function_fid`, `preprocess_eli5_validation_function_fid`, `preprocess_eli5_batch_fid`

- For Copy Mechanism: `preprocess_eli5_function_cmsks`, `preprocess_eli5_validation_function_cmsks`, `preprocess_eli5_batch_cmsks`.

- For Copy Mechanism + ANS_oracle: `preprocess_eli5_function_cmsks_oracle`, `preprocess_eli5_validation_function_cmsks_oracle`, `preprocess_eli5_batch_cmsks_oracle`.

In `eli5_oracle.py` we have ANS_oracle with BART and FiD:
- For BART + ANS_oracle: `preprocess_eli5_validation_function`, `preprocess_eli5_function`, `preprocess_eli5_batch`.

- For FiD + ANS_oracle: `encode_passages`, `preprocess_eli5_function_fid`, `preprocess_eli5_validation_function_fid`, `preprocess_eli5_batch_fid`

In `eli5_oraclekg.py` we have KG_oracle with BART and FiD. Function names are the same as in `eli5_oracle.py`


In `fid_kg.py` we have n-hop KG with FiD.



## Trainers
A seq2seq trainer.


## Scripts

For OpenIE KG construction:

**Note that `neuralcoref` is not compatible with `spacy` version 3. So I recommend using another environement for KG construction before we improve the code **

requirements:
- marisa_trie
- neuralcoref
- spacy==2.1.0 (and download the model with `python -m spacy download en_core_web_sm`)


1. To extract OpenIE triples from text and save as json files, run `primeqa/examples/lfqa_kb/run_scripts/run_openie.sh`. Needs to specify the start and end index of the data examples (I did 5000 at the time). The code is in `primeqa/primeqa/lfqa_kb/scripts/run_openie.py`
2. To merge triples into graphs and save as pkl files., run `/dccstor/myu/primeqa/examples/lfqa_kb/run_scripts/postprocess_openie.sh`.
3. To merge graphs of dev and train set, use `/dccstor/myu/primeqa/primeqa/lfqa_kb/scripts/merge_kg.py`

