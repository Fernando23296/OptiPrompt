import os
import json
import shutil
from qa_utils import load_jsonl
import pandas as pd

val_trusted_list = ["True", "False"]
val_target_list = ["Yes","No"]
# Don't forget to put a space at the end
val_question_list = ['']
SEQUENCES = False
TEMPLATES = False


def get_dic_questions(case):
    
    lama_relations_path = 'prompts/LAMA_relations_'+str(case)+'.jsonl'
    lama_relations = load_jsonl(lama_relations_path)
    #print("LAMA CARGADA")
    #print(lama_relations)
    questions = []
    for i in lama_relations:
        #Possible to add the fact
        ##print(i['template'])
        questions.append(i['template'])

    dict_questions = {key: [] for key in questions}
    ##print("*"*10)
    ##print("dict_questions")
    ##print(dict_questions)
    ##print("*"*10)
    return dict_questions

def extract_acc(dictionary_values, case):
    #print("DICT VALUES")
    #print(dictionary_values)
    # getting dict with questions as keys, and empty of values
    dict_questions = get_dic_questions(case)
    for d in dictionary_values:
        for i in dictionary_values[d]:
            data  = [i[0][2], i[0][3], i[0][4],d]
            dict_questions[i[0][0]].append(data)

    #print(dict_questions)
    final_list = []
    c = 0
    for dict_q in dict_questions:
        acc = dict_questions[dict_q][0][0]/(dict_questions[dict_q][0][1]*2)
        final_list.append([dict_q, acc, dict_questions[dict_q][0][2],  dict_questions[dict_q][0][3]])
        c +=1
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
        #print("*"*10)
        #print(val_question)
        fname = str(val_trusted_list[i]) + "_" + str(val_target_list[i])
        os.system(f"python main_qa.py --trusted {val_trusted_list[i]}  --target {val_target_list[i]} --question '{val_question}' --sequences '{SEQUENCES}' --templates '{TEMPLATES}'")
        os.system(f"python code/run_eval_prompts.py --model_name facebook/opt-350m --prompt_file prompts/LAMA_relations.jsonl --trusted  {val_trusted_list[i]} --filename {fname}")
        
        src_folder = r"prompts/"
        dst_folder = r"prompts/"

        # file names
        #print("FLAG 1")
        src_file = src_folder + "LAMA_relations.jsonl"
        val_q = val_question.replace(" ", "_")
        #print("VAL Q", val_q)
        name_pre = str(val_trusted_list[i] + "_" + val_target_list[i] + "_" + val_q)
        #print("FLAG 2")
        #print("NAME PRE", name_pre)
        #prev
        #dst_file = dst_folder + "LAMA_relations_" + name_pre + ".jsonl"
        
        dst_file =  dst_folder +  "LAMA_relations_" + name_pre + ".jsonl"
        
        #print("DST_FILE")
        #print(dst_file)

        shutil.copyfile(src_file, dst_file)

        
        

for i in range(len(val_trusted_list)):
    for val_question in val_question_list:
        # file names
        fname_df = str(val_trusted_list[i]) + "_" + str(val_target_list[i])
        #src_file = src_folder + "LAMA_relations.jsonl"
        val_q = val_question.replace(" ", "_")
        name_pre = str(val_trusted_list[i] + "_" + val_target_list[i] + "_" + val_q)
        dst_file = dst_folder + "LAMA_relations_" + name_pre
        
        #shutil.copyfile(src_file, dst_file)

        # reading the data from the file
        with open('results/'+fname_df+".txt") as f:
            data = f.read()
            
        # reconstructing the data as a dictionary
        final_quest = json.loads(data)
        #print("FINAL QUEST")
        #print(final_quest)
        #print("*!"*20)
        #print("NAME")
        #print(name_pre)
        final_list = extract_acc(final_quest, name_pre)
        df_final_list = pd.DataFrame(final_list, columns = ['question', 'acc', 'target','fact'])
        fname_df_final = fname_df + val_question + "_dataframe.csv"
        df_final_list.to_csv(fname_df_final)
        df_final_list.to_csv("results/"+fname_df_final)
        #print(df_final_list.head())

# lo de arriba funciona bien
# Merging and obtaining final accuracies
for val_question in val_question_list:

    # Obtaining: True_yes_dataframe.csv
    positive_side  = val_trusted_list[0] + "_" + val_target_list[0] + val_question + "_dataframe.csv"
    positive_side_df = pd.read_csv("results/"+positive_side)

    # Obtaining: False_no_dataframe.csv
    negative_side  = val_trusted_list[1] + "_" + val_target_list[1] + val_question + "_dataframe.csv"
    negative_side_df = pd.read_csv("results/"+negative_side)

    df_master = pd.merge(positive_side_df, negative_side_df, on='question')
    #df_master.head()
    df_master['accuracy'] = df_master['acc_x'] + df_master['acc_y']
    #df_master.head()
    df_master = df_master.sort_values(by=['fact_x'])
    #df_master.head(20)

    # is the same to use fact_x and fact_y
    df_group = df_master.groupby(['fact_x']).max()
    df_group.head()

    df_final = df_group[['question', 'accuracy']]
    #print(df_final.head())

    df_final.to_csv("results/"+"question_"+val_question+".csv")
