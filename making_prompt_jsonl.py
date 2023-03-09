'''
This file generates files for prompt changing the question and the status of trusted.

RETURN:
prompt/LAMA_relations.jsonl
'''
import os
import json
import random
import pandas as pd


def making_prompt_jsonl(STATUS, QUESTION_TYPE, DEMONSTRATION = False, DEMONSTRATION_EXTRA = False, SUPPORT = False):

    #STATUS = "trusted"
    # question without ":"
    #QUESTION_TYPE = "The answer is"
    triplet_list_true = [["[TA]","[TB]"],["[TC]","[TD]"],["[TE]","[TF]"],["[TG]","[TH]"],["[TI]","[TJ]"]]
    triplet_list_fake = [["[FA]","[FB]"],["[FC]","[FD]"],["[FE]","[FF]"],["[FG]","[FH]"],["[FI]","[FJ]"]]
    with open('extra/templates.txt') as f:
        lines = f.readlines()

    selected_templates_names = [i.split()[1] for i in lines if len(i) > 1 and i.split()[1][0] == 'P']

    selected_templates = [i.split()[1]+'.jsonl' for i in lines if len(i) > 1 and i.split()[1][0] == 'P']
    #print(selected_templates)

    directory_path = 'data/LAMA/data/TREx/'
    directory_files = os.listdir(directory_path)

    files = [i for i in directory_files if i[-5:] == 'jsonl']

    templates = [file for file in files if file in selected_templates]


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

    def clean_question(question):
        if '\n' in question:
            question.split('\n')[0]
        return question


    def joiner(list_masked, list_obj, list_sub, file_name, question, STATUS, DEMONSTRATION, DEMONSTRATION_EXTRA):
        n = []
        #print("List masked:", list_masked)
        #print("List obj:", list_obj)
        
        def generate_prompt(fname, sentence, triplet, state):
            
            new = sentence.replace("[X]", "["+fname + "_"+ triplet[0][1:])
            new = new.replace("[Y]", "["+fname + "_"+ triplet[1][1:])
            # aqui se cambia si se quiere poner o no el dos puntos
            # no question for extra sentences
            new = new[:-1]+" "+state
            return new
            
            
        for i in range(len(list_masked)):
            # index with -1 in order to get only the question
            list_masked[i] = clean_question(list_masked[i])
            if DEMONSTRATION == False:
                new = list_masked[i].replace("[X]", "[W]")
                new = new.replace("[Y]", "[X]")
                # aqui se cambia si se quiere poner o no el dos puntos
                new = new[:-1]+" "+question+"[Y]"
                n.append([file_name, new, STATUS])
                # Generating fake sentences only varying the object field
                # Dropping duplicates
                '''
                set_obj = list(set(list_obj))

                # Obtaining a random unverified record
                while True:
                    rn = random.randint(0, len(set_obj)-1)
                    if set_obj[rn] != list_obj[ii]:
                        fake = list_masked[i].replace("[X]", list_sub[ii])
                        fake = fake.replace("[Y]", set_obj[rn])
                        n.append([file_name, fake,'unverified'])
                        break
                        '''
            if DEMONSTRATION:
                seq, val_random = DEMONSTRATION.split()
                val_random = val_random.lower()
                new_prompt = ""
                if val_random == "normal":
                    for d in seq:
                        c_true = 0
                        c_fake = 0
                        if d == "T" or d == "t":
                            state = DEMONSTRATION_EXTRA + "Yes. "
                            new_prompt += generate_prompt(file_name, list_masked[i], triplet_list_true[c_true], state)
                            c_true += 1
                        else:
                            state = DEMONSTRATION_EXTRA + "No. "
                            new_prompt += generate_prompt(file_name, list_masked[i], triplet_list_fake[c_fake], state)
                            c_fake += 1
                            
                if val_random == "random":
                    
                    for d in seq:
                        c_true = 0
                        c_fake = 0
                        
                        while True:
                                rn = random.randint(0, len(list_masked)-1)
                                if list_masked[rn] != list_masked[i]:
                                    ii = rn
                                    break
                                    
                        if d == "T" or d == "t":
                            state = DEMONSTRATION_EXTRA + "Yes. "
                            new_prompt += generate_prompt(file_name, list_masked[ii],triplet_list_true[c_true], state)
                            c_true += 1
                            
                        else:
                            state = DEMONSTRATION_EXTRA + "No. "
                            new_prompt += generate_prompt(file_name, list_masked[ii],triplet_list_fake[c_fake], state)
                            c_fake += 1
                    
                list_masked[i] = new_prompt + list_masked[i]
                new = list_masked[i].replace("[X]", "[W]")
                new = new.replace("[Y]", "[X]")
                # aqui se cambia si se quiere poner o no el dos puntos
                new = new[:-1]+" "+question+"[Y]"
                n.append([file_name, new, STATUS])
                
        return n


    with open('relation-paraphrases.txt') as f:
        lines = f.readlines()

    indexes = [i for i, val in enumerate(lines) if val[:2] == "*P"]


    aux_dict = {}
    for i in range(len(indexes)-1):
        aux_dict[lines[indexes[i]]] = lines[indexes[i]: indexes[i+1]]
    aux_dict[lines[indexes[-1]]] = lines[indexes[-1]:]


    questions = {k.split()[0][1:]: v for k, v in aux_dict.items()}


    lama_relations = load_jsonl("LAMA_relations_original.jsonl")
    selected_lama_relations = [i for i in lama_relations if i['relation'] in selected_templates_names]

    def jsonl_builder(template):
        for i in template:
            i.extend(next([item["label"], item["description"], item["type"]] for item in selected_lama_relations if item["relation"] == i[0]))
        return template


    def clean_rows(lists):
        # Cleaning the prefix
        return [i[3:] for i in lists if i[0:2] == 'Q:']    


    qs_ = [QUESTION_TYPE]
    for question in qs_:
        c = 0
        pre_df_master = []
        for k, v in questions.items():
            for t in templates:
                #print(t)
                name = t.split(".")[0]
                if k == name:
                    t_ = load_jsonl("data/LAMA/data/TREx/"+t)
                    obj, sub = get_data(t_)
                    cr = clean_rows(v[2:])
                    pre_df = jsonl_builder(joiner(cr, obj, sub, name, question, STATUS, DEMONSTRATION, DEMONSTRATION_EXTRA))
                    pre_df_master.extend(pre_df)
                    #print(pre_df)
        df = pd.DataFrame(pre_df_master, columns = ["relation", "template", "status", "label", "description", "type"])
        
        path_files = os.path.join("prompts/", "LAMA_relations.jsonl")
        
        if SUPPORT:
            PATH = "prompts/LAMA_relations_support.jsonl"
            df.to_json(PATH, orient='records', lines=True)
        
        else:
            PATH = "prompts/LAMA_relations.jsonl"
            df.to_json(PATH, orient='records', lines=True)
        



