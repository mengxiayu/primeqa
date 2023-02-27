import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import sys

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text2words(text):
    words = set()
    for x in text.split():
        w = lemmatizer.lemmatize(x.lower())
        if w not in stop_words and w not in string.punctuation:
            words.add(w)   
    return words

def compute_scores(data):
    question_words = text2words(data["input"])
    
    # words in top 3 passages
    top_passage_words = []
    for p in data["passages"][:3]:
        words = text2words(p['text'])
        top_passage_words.append(words)

    # kg_vocab scores excluding top-3 passages
    max_psg_score = -1
    max_kg_vocab_score = -1

    # for each answer
    for output in data["output"]:
        answer = output["answer"]
        answer_words = text2words(answer) - question_words
        kg_vocab = set(output["kg_vocab"])

        # kg_triples 
        kg_triples = [x for x in output["kg_triples"] if x["count"] >= 2]
        kg_triple_words = set()
        cnt_length = 0
        for t in kg_triples:
            text = t["text"]
            if cnt_length > 250:
                break
            cnt_length += text.count(' ')
            kg_triple_words.update(text2words(text))

        # kg sentences
        kg_sentences = [x for x in output["kg_sentences"] if x["count"] >= 2]
        kg_sent_words = set()
        cnt_length = 0
        for t in kg_sentences:
            text = t["sentence"]
            if cnt_length > 250:
                break
            cnt_length += text.count(' ')
            kg_sent_words.update(text2words(text))           
        # top 1,2,3 passages true positive
        tp_top1_psg_words = top_passage_words[0] & answer_words
        tp_top2_psg_words = (top_passage_words[0] | top_passage_words[1]) & answer_words
        tp_top3_psg_words = (top_passage_words[0] | top_passage_words[1] | top_passage_words[2]) & answer_words
        
        tp_kg_vocab_words = kg_vocab & answer_words
        tp_kg_triple_words = kg_triple_words & answer_words
        tp_kg_sent_words = kg_sent_words & answer_words

        # recall
        psg_score = len(tp_top3_psg_words) / len(answer_words) if answer_words else 0
        kg_vocab_score = len(tp_kg_vocab_words - tp_top3_psg_words) / len(answer_words) if answer_words else 0
        kg_triple_score = len(tp_kg_triple_words - tp_top3_psg_words) / len(answer_words) if answer_words else 0
        kg_sent_score = len(tp_kg_triple_words - tp_top3_psg_words) / len(answer_words) if answer_words else 0
  
        output["meta"]["psg_score"] = psg_score
        output["meta"]["kg_vocab_score"] = kg_vocab_score
        output["meta"]["kg_triple_score"] = kg_triple_score
        output["meta"]["kg_sentences_score"] = kg_sent_score
        output["meta"]["psg_prec"] = len(tp_top3_psg_words) / len(top_passage_words[0] | top_passage_words[1] | top_passage_words[2])
        output["meta"]["kg_vocab_prec"] = len(tp_kg_vocab_words) / len(kg_vocab) if kg_vocab else 0
        output["meta"]["kg_triple_prec"] = len(tp_kg_triple_words) / len(kg_triple_words) if kg_triple_words else 0
        output["meta"]["kg_sent_prec"] = len(tp_kg_sent_words) / len(kg_sent_words) if kg_sent_words else 0
        # precision
        if psg_score > max_psg_score:
            max_psg_score = psg_score
            max_kg_vocab_score = kg_vocab_score
            max_kg_triple_score = kg_triple_score
            max_kg_sent_score = kg_sent_score
            
            psg_prec = output["meta"]["psg_prec"]
            kg_vocab_prec = output["meta"]["kg_vocab_prec"]
            kg_triple_prec = output["meta"]["kg_triple_prec"]
            kg_sent_prec = output["meta"]["kg_sent_prec"]
        

    # print(max_score, words)
    data['scores'] = {}
    data['scores']['max_psg_score'] = max_psg_score
    data['scores']['max_kg_vocab_score'] = max_kg_vocab_score
    data['scores']['max_kg_triple_score'] = max_kg_triple_score
    data['scores']['max_kg_sentences_score'] = max_kg_sent_score
    data['scores']['max_psg_prec'] = psg_prec
    data['scores']['max_kg_vocab_prec'] = kg_vocab_prec
    data['scores']['max_kg_triple_prec'] = kg_triple_prec
    data['scores']['max_kg_sentences_prec'] = kg_sent_prec
    return data

def main():

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    writer = open(output_file, 'w')
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            data = compute_scores(data)
            # print out updated data file
            writer.write((json.dumps(data) + "\n"))

if __name__ == '__main__':
    main()
