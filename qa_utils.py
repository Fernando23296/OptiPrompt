import json


def load_jsonl(file_name):
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
    all_dicts = []
    for json_str in json_list:
        all_dicts.append(json.loads(json_str))
        
    return all_dicts