import os
import json
import random
import pandas as pd
import pprint
import pickle
from qa_utils import load_jsonl
from making_prompt_jsonl import making_prompt_jsonl

def making_multiple_prompt_jsonl(STATUS, TARGET, QUESTION_TYPE, SEQUENCE, TEMPLATES, DEMONSTRATION_EXTRA):
    '''
    # three true facts before the fact to be tested
    SEQUENCE = "ttt"
    # normal or random
    # normal means same template
    # random will shuffle
    TEMPLATES = "normal"
    '''
    if TARGET == "Yes":
        ANTI_TARGET = "No"

    elif TARGET == "YES":
        ANTI_TARGET = "NO"

    else:
        ANTI_TARGET = "no"
    # NEEDS 'data/filtered_LAMA'
    def obtain_true_values(filename):
        '''
        Obtain list of true pairs of values
        '''
        true_list = []
        for i in filename:
            true_list.append([i["sub_label"], i["sub_label_2"]])
        return true_list


    def obtain_false_values(filename):
        '''
        Obtain list of true pairs of values
        '''
        fake_list = []
        list_sub_label_2 = [s["sub_label_2"] for s in filename]
        for i in filename:
            true_sub_label = i["sub_label"]
            true_sub_label_2 = i["sub_label_2"]
            while True:
                rn = random.randint(0, len(list_sub_label_2)-1)
                if list_sub_label_2[rn] != true_sub_label_2:
                    #Untrusted fact
                    fake_list.append([true_sub_label, list_sub_label_2[rn]])
                    break
            
        return fake_list


    directory_path = 'data/filtered_LAMA'
    directory_files = os.listdir(directory_path)
    files = [i for i in directory_files if i[-5:] == 'jsonl']

    dict_true = {}
    for file in files:
        f = load_jsonl(directory_path + "/" + file)
        dict_true[file[:-6]] = obtain_true_values(f)

    '''
    #Saving dictionary
    tr = open("extra/true_fact.txt","w")
    tr.write( str(dict_true) )
    tr.close()
    '''
    with open("extra/true_fact.txt", 'wb') as fp:
        pickle.dump(dict_true, fp)

    dict_fake = {}
    for file in files:
        f = load_jsonl(directory_path + "/" + file)
        dict_fake[file[:-6]] = obtain_false_values(f)

    '''
    #Saving dictionary
    tr = open("extra/fake_fact.txt","w")
    tr.write( str(dict_fake) )
    tr.close()
    '''
    with open("extra/fake_fact.txt", 'wb') as fp:
        pickle.dump(dict_fake, fp)


    def question_filler(question, answers_dict, status):
        aux = []
        for ans in answers_dict:
            ans1 = ans[0]
            ans2 = ans[1]
            #print(ans1)
            #print(ans2)
            q = question.replace('[W]',ans1)
            q = q.replace('[X]',ans2)
            q = q.replace('[Y]',status)
            aux.append(q)
        return aux

    making_prompt_jsonl(STATUS, "", SUPPORT = True)

    prompt = load_jsonl("prompts/LAMA_relations_support.jsonl")

    #Creating dictionary questions
    dict_questions = {}
    for p in prompt:
        dict_questions[p["relation"]] = []

    #obtaining questions
    for p in prompt:
        dict_questions[p["relation"]].append(p["template"])


    #Creating dictionary questions
    true_facts = {}
    for p in prompt:
        true_facts[p["relation"]] = []

    for file in dict_questions:
        for question in dict_questions[file]:
            true_facts[file].extend(question_filler(question, dict_true[file], TARGET))


    
    #Creating dictionary questions
    fake_facts = {}
    for p in prompt:
        fake_facts[p["relation"]] = []

    for file in dict_questions:
        for question in dict_questions[file]:
            fake_facts[file].extend(question_filler(question, dict_fake[file], ANTI_TARGET))




    DEMONSTRATION = str(SEQUENCE + " " + TEMPLATES)
    making_prompt_jsonl(STATUS, QUESTION_TYPE, DEMONSTRATION, DEMONSTRATION_EXTRA)