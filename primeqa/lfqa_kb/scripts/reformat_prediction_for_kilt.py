import json
import argparse

# reformat prediction file so that it can be directly used as input file for KILT evaluation

def reformat_prediction_file(kilt_file, prediction_file):
    assert prediction_file.endswith(".json") and kilt_file.endswith(".json")
    with open(prediction_file, 'r') as f:
        predictions = json.loads(f.read())
    id2pred = {}
    for item in predictions:
        id2pred[item["id"]] = item["prediction_text"] 
    assert len(id2pred) == len(predictions)

    with open(prediction_file[:-5] + "_reformat.json", 'w') as fw:
        with open(kilt_file, 'r') as fr:
            for line in fr:
                result = {}
                data = json.loads(line)
                result["id"] = data["id"]
                result["input"] = data["input"]

                pred = id2pred[data["id"]]
                result["output"] = [{"answer": pred}]
                fw.write(json.dumps(result) + "\n")
    print("reformatted prediction file saved.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kilt_file", help="Gold KILT file")
    parser.add_argument("--pred_file", help="Gold KILT file")
    args = parser.parse_args()
    reformat_prediction_file(args.kilt_file, args.pred_file)
