import os
import json
import shutil
from qa_utils import load_jsonl
import pandas as pd

val_trusted_list = ["True", "False"]
val_target_list = ["yes","no"]
val_question_list = ['TELL ME']


def get_dic_questions(case):
    
    lama_relations_path = 'prompts/LAMA_relations_'+str(case)+'.jsonl'
    lama_relations = load_jsonl(lama_relations_path)
    print("LAMA CARGADA")
    print(lama_relations)
    questions = []
    for i in lama_relations:
        #Possible to add the fact
        #print(i['template'])
        questions.append(i['template'])

    dict_questions = {key: [] for key in questions}
    #print("*"*10)
    #print("dict_questions")
    #print(dict_questions)
    #print("*"*10)
    return dict_questions

def extract_acc(dictionary_values, case):
    print("DICT VALUES")
    print(dictionary_values)
    # getting dict with questions as keys, and empty of values
    dict_questions = get_dic_questions(case)
    for d in dictionary_values:
        #print(d)
        #print("*"*10)
        #print("D VALUES")
        #print(d)
        #print("*"*10)
        for i in dictionary_values[d]:
            print("*"*10)
            print("VALUEI")
            print(i)
            print("*"*10)
            data  = [i[0][2], i[0][3], i[0][4],d]
            print("*"*10)
            print("VALUEDATA")
            print(data)
            print("*"*10)
            dict_questions[i[0][0]].append(data)

    print("DICT QUESTIONS")
    print(dict_questions)
    final_list = []
    #print("="*10)
    #print(len(dict_questions))
    #print(dict_questions)
    c = 0
    for dict_q in dict_questions:
        #print(dict_questions[dict_q][0][3])
        #print("ACAAAAAAAAAAAAAAAAA")
        #print("/"*10)
        #print(dict_q)
        #print(dict_questions[dict_q])
        #print("/"*10)
        #print(dict_questions[dict_q][0])
        #print("/"*10)
        #print(dict_questions[dict_q][0][0])
        #print("/"*10)
        #print(dict_questions[dict_q][0][1]*2)
        acc = dict_questions[dict_q][0][0]/(dict_questions[dict_q][0][1]*2)
        #print("/acc"*10)
        #print(acc)
        final_list.append([dict_q, acc, dict_questions[dict_q][0][2],  dict_questions[dict_q][0][3]])
        c +=1
        #print(c)
    return final_list


#writing the common vocab
with open('common_vocab_binary.txt', 'w') as f:
    for line in val_target_list:
        f.write(line)
        f.write('\n')


val_trusted_res = {}
for i in range(len(val_trusted_list)):
    val_question_res = {}
    for val_question in val_question_list:
        print("*"*10)
        print(val_question)
        fname = str(val_trusted_list[i]) + "_" + str(val_target_list[i])
        os.system(f"python main_qa.py --trusted {val_trusted_list[i]}  --target {val_target_list[i]} --question '{val_question}'" )
        os.system(f"python code/run_eval_prompts.py --model_name facebook/opt-350m --prompt_file prompts/LAMA_relations.jsonl --trusted  {val_trusted_list[i]} --filename {fname}")
        
        src_folder = r"prompts/"
        dst_folder = r"prompts/"

        # file names
        src_file = src_folder + "LAMA_relations.jsonl"
        val_q = val_question.replace(" ", "_")
        name_pre = str(val_trusted_list[i] + "_" + val_target_list[i] + "_" + val_q)
        dst_file = dst_folder + "LAMA_relations_" + name_pre + ".jsonl"

        shutil.copyfile(src_file, dst_file)

        
  
        # reading the data from the file
        with open('results/'+fname+".txt") as f:
            data = f.read()
            
        # reconstructing the data as a dictionary
        final_quest = json.loads(data)
        print("FINAL QUEST")
        print(final_quest)
        print("*!"*20)
        print("NAME")
        print(name_pre)
        final_list = extract_acc(final_quest, name_pre)
        df_final_list = pd.DataFrame(final_list, columns = ['question', 'acc', 'target','fact'])

        print(df_final_list.head())
# calculating the values from files

'''
val_trusted_res = {}
for val_trusted in val_trusted_list:
    val_question_res = {}
    for val_question in val_question_list:
        os.system(f"python main_qa.py --trusted {val_trusted}  --target {target} --question {val_question}")
        os.system(f"python code/run_eval_prompts.py --model_name facebook/opt-350m --prompt_file prompts/LAMA_relations.jsonl --trusted  {val_trusted}")

    
        with open('results_acc/new.txt') as f:
            data = f.read()

        print("DATA CAPTURED")
        print(data)
        val_question_res[val_question] = data
    
    val_trusted_res[val_trusted] = val_question_res

print(val_trusted_res)
'''
