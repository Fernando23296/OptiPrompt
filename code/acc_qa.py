import os
import json

def get_max_binary(record):
    max_binary = record['topk'][0]['token']
    return max_binary

def count_binary(json_file, target):
    count_target = 0
    for i in range(len(json_file)):
        if get_max_binary(json_file[i]) == target:
            count_target += 1
    total = len(json_file)
    return count_target, total

directory_path = "output/LAMA_relations/facebook_opt-350m/"
directory_files = os.listdir(directory_path)


def load_jsonl(file_name):
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
    all_dicts = []
    for json_str in json_list:
        all_dicts.append(json.loads(json_str))
        
    return all_dicts


def count_results():
    count_results = []
    for df in directory_files:
        jsonl_file = load_jsonl(directory_path + df)
        
        # Precaution: works when the whole file have the same target
        target = jsonl_file[0]['obj_label']
        
        count_target, total = count_binary(jsonl_file, target)
        
        count_results.append([df, count_target, total, target])
