'''
This file change the obj_label from LAMA jsonl files to TRUSTED_VALUE"

RETURN:
Files in filtered_LAMA
'''
import os
import pprint
import json
import jsonlines


def filtered_LAMA_QA(TRUSTED_VALUE):

    #TRUSTED_VALUE = "Yes"

    directory_path = 'data/LAMA/data/TREx/'

    directory_path_prompt = 'prompts/'

    directory_files = os.listdir(directory_path)
    files = [i for i in directory_files if i[-5:] == 'jsonl']

    with open('extra/templates.txt') as f:
        lines = f.readlines()

    selected_templates_names = [i.split()[1] for i in lines if len(i) > 1 and i.split()[1][0] == 'P']
    selected_templates = [i.split()[1]+'.jsonl' for i in lines if len(i) > 1 and i.split()[1][0] == 'P']
    #print(selected_templates)


    # Directory where filtered files will be
    directory_path_filtered = 'data/filtered_LAMA'
    if not os.path.exists(directory_path_filtered):
        os.makedirs(directory_path_filtered)

    def clean_rows(lists):
        # Cleaning the prefix
        return [i[3:] for i in lists if i[0:2] == 'Q:']    


    def load_jsonl(file_name):
        with open(file_name, 'r') as json_file:
            json_list = list(json_file)
        all_dicts = []
        for json_str in json_list:
            all_dicts.append(json.loads(json_str))
            
        return all_dicts



    def get_data(list_dicts):
        obj_data = []
        sub_data = []
        for i in list_dicts:
            obj_data.append(i['obj_label'])
            sub_data.append(i['sub_label'])
        return obj_data, sub_data




    for t in selected_templates:
        #print(t)
        list_jsonl = []
        a = load_jsonl(directory_path + t)
        path_files = os.path.join(directory_path_filtered, t)
        with jsonlines.open(path_files, 'w') as writer:
            for i in a:
                # Creating new keys for QA mode
                i["sub_label_2"] = i["obj_label"]
                
                # If the context is TRUSTED then the obj_label will be Yes
                i["obj_label"] = TRUSTED_VALUE
                list_jsonl.append(i)
            writer.write_all(list_jsonl)
        




