import json
import csv
with open('./dev-v2.0.json', "r") as reader:
        input_data = json.load(reader)["data"]

gt = {}

for entry in input_data:
    for paragraph in entry["paragraphs"]:
        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            answerable = qa["is_impossible"]
            gt[qas_id] = answerable

tp = 0
tn = 0
fp = 0
fn = 0
total = 0
with open('classification.csv', "r") as f:
    predictions = csv.reader(f)
    for row in predictions:
        id = row[0]
        guess = int(row[1])
        if gt[id] == guess and guess == 0:
            tn += 1
        elif gt[id] == guess and guess == 1:
            tp += 1
        elif gt[id] != guess and guess == 1:
            fp += 1
        elif gt[id] != guess and guess == 0:
            fn += 1
        total += 1

print("tp:{}, tn:{}, fp:{}, fn:{}, total:{}".format(tp,tn,fp,fn,total))