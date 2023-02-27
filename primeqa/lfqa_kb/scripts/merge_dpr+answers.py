import gzip
import json
import sys
from tqdm import tqdm

def parse_passages(data_file, answer_file, output_file):
    data_lines = []
    with open (data_file, 'r') as f:
        for line in f:
            data_lines.append(json.loads(line.strip()))

    i = 0
    with open (answer_file, 'r') as f:
        for line in f:
            answer_data = json.loads(line.strip())
            if answer_data['id'] != data_lines[i]['id']:
                print("Error! mismatching IDs! " + answer_data['id'] + " " + str(i) + " " + data_lines[i]['id'])
                return
            data_lines[i]['output'] = answer_data['output']
            i+= 1

    with open(output_file, 'w') as fout:       # 4. fewer bytes (i.e. gzip)
        for line in data_lines:
            json_str = json.dumps(line) + "\n"               # 2. string (i.e. JSON)
            # json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)
            #fout.write(json_bytes) 
            fout.write(json_str)


parse_passages(sys.argv[1], sys.argv[2], sys.argv[3])                      

# data_file = "/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg/eli5-train-kilt-dpr-kg-34.json"
# data_lines = []
# with open (data_file, 'r') as f:
#     examples  = f.readlines()
#     for ex in tqdm(examples, total=len(examples)):
#         data_lines.append(json.loads(ex))
       