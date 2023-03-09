import json
import os
import re
from tqdm import tqdm
import sys
import logging
import random
import numpy as np
import json
import pickle
  

logger = logging.getLogger(__name__)

#CHANGE
# for evaluation
#list_pre_acc = []

#CHANGE


def load_jsonl(file_name):
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
    all_dicts = []
    for json_str in json_list:
        all_dicts.append(json.loads(json_str))
        
    return all_dicts

def get_max_binary(record):
    max_binary = record['topk'][0]['token']
    return max_binary

# CHANGE
def count_binary(json_file, target):
    count_target = 0
    count_otro = 0
    for i in range(len(json_file)):
        if get_max_binary(json_file[i]) == target:
            count_target += 1
        else:
            count_otro += 1
    
    total = len(json_file)
    return count_target, total


def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

# NEW
def multiple_prompts_filler(dict_facts, subject_label, sub_label_2, templates, record_):
    record = []
    options = dict_facts[templates]
    while True:
        rn = random.randint(0, len(options)-1)
        if subject_label != options[rn][0] and options[rn][0] not in record_:
            return options[rn][0], options[rn][1]


def parse_template(template, subject_label, subject_label_2, obj_label, obj_data, trusted, object_label='[MASK]'):
    
    
    SUBJ_SYMBOL_W = "[W]"
    SUBJ_SYMBOL_X = "[X]"
    OBJ_SYMBOL = "[Y]"

    pattern = re.compile(r'\[[^\]]*\]')
    new_masks = pattern.findall(template)
    #print("?"*20)
    #print("NEW MASKS")
    #print(new_masks)

    with open("extra/true_fact.txt", 'rb') as fp:
        true_facts = pickle.load(fp)

    with open("extra/fake_fact.txt", 'rb') as fp:
        fake_facts = pickle.load(fp)


    #print("TRUEFACTS")
    #print(true_facts)

    #print("FAKEFACTS")
    #print(fake_facts)
    
    # -3 because these will be later replaced
    #print("arctic")
    #print(len(new_masks[:-3]))
    record_ = []
    for i in range(0,len(new_masks[:-3]),2):
        sp = new_masks[i].split("_")
        #template
        temp = sp[0][1:]
        #state, whether true or fake
        state = sp[1][:1]
        #print("ESTOOOOO")
        #print(state)
        
        if state == "T":
            new_first, new_second = multiple_prompts_filler(true_facts, subject_label, subject_label_2, temp, record_)
        else:
            new_first, new_second = multiple_prompts_filler(fake_facts, subject_label, subject_label_2, temp, record_)

        # reccord helps to ensure not double values in same prompt
        record_.append(new_first)
        #print("AUXILIAR")
        #print("i", i)
        #print("NEW FIRST", new_first)
        #print("NEW SECOND", new_second)
        template = template.replace(new_masks[i], new_first)
        template = template.replace(new_masks[i+1], new_second)
    

    template = template.replace(SUBJ_SYMBOL_W, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    #CHANGE
    # Binary
    if trusted:
        #trusted fact
        template = template.replace(SUBJ_SYMBOL_X, subject_label_2)
        
    else:
        while True:
            rn = random.randint(0, len(obj_data)-1)
            if obj_data[rn] != object_label:
                #Untrusted fact
                template = template.replace(SUBJ_SYMBOL_X, obj_data[rn])
                break

    #print("MIRAAAAAAAAAAAAAAAAAAAA")
    #print(template)
    return [template]

def convert_tokens_to_string(tokens):
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

def get_relation_meta(prompt_file, relation_name):
    relations = load_file(prompt_file)
    for relation in relations:
        if relation['relation'] == relation_name:
            return relation
    raise ValueError('Relation info %s not found in file %s'%(relation_name, prompt_file))

# CHANGE
def count_relation_meta(prompt_file, relation_name):
    
    relations = load_file(prompt_file)
    c = 0
    l_templates = []
    for relation in relations:
        
        if relation['relation'] == relation_name:
            c += 1
            l_templates.append(relation)

    return c, l_templates
    


def batchify(data, batch_size):
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    
    c = 0
    for sample in data:
        input_sentences = sample['input_sentences']
        current_samples_batch.append(sample)
        current_sentences_batches.append(input_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches


def save_model(model, args):
    logger.info('Saving model...')
    model_to_save = model.mlm_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)

def output_result(result, eval_loss):
    logger.info('* Evaluation result *')
    cor = 0
    tot = 0
    macro = 0.0
    loss = 0.0
    for rel in result:
        cor_, tot_, avg_, loss_ = result[rel]
        cor += cor_
        tot += tot_
        macro += avg_
        loss_ /= tot_
        loss += loss_
        logger.info('%s\t%.5f\t%d\t%d\t%.5f' % (rel, avg_, cor_, tot_, loss_))
    macro = cor / tot if tot > 0 else 0.0
    micro = macro / len(result) if len(result) > 0 else 0.0
    logger.info('Macro avg: %.5f' % macro)
    logger.info('Micro avg: %.5f, Eval_loss: %.5f, Eval_loss (common vocab): %.5f' %(micro, eval_loss / tot, loss / len(result) if len(result) > 0 else 0.0))
    sys.stdout.flush()
    return micro, macro

def evaluate(model, samples_batches, sentences_batches, template, filter_indices=None, index_list=None, output_topk=None):
    #CHANGE
    list_pre_acc = []
    
    vocab_to_common_vocab = None
    if index_list is not None:
        vocab_to_common_vocab = {}
        for cid, idx in enumerate(index_list):
            vocab_to_common_vocab[idx] = cid

    cor_all = 0
    tot_all = 0
    result = {}
    list_of_predictions = {}
    eval_loss = 0.0
    common_eval_loss = 0.0
    for i in tqdm(range(len(samples_batches))):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]

        log_probs, cor_b, tot_b, pred_b, topk_preds, loss, common_vocab_loss = model.run_batch(sentences_b, samples_b, training=False, filter_indices=filter_indices, index_list=index_list, vocab_to_common_vocab=vocab_to_common_vocab)
        cor_all += cor_b
        tot_all += tot_b

        for pred, sample, topk, vocab_loss in zip(pred_b, samples_b, topk_preds, common_vocab_loss):
            rel = sample['predicate_id']
            if rel not in result:
                result[rel] = (0, 0, 0, 0.0)
                list_of_predictions[rel] = []
            cor, tot, _, rel_tot_loss = result[rel]
            tot += 1
            cor += pred
            rel_tot_loss += vocab_loss
            result[rel] = (cor, tot, cor / tot if tot > 0 else 0.0, rel_tot_loss)
            list_of_predictions[rel].append({
                'uuid': sample['uuid'],
                'relation': sample['predicate_id'],
                'sub_label': sample['sub_label'],
                'sub_label_2': sample['sub_label_2'],
                'obj_label': sample['obj_label'],
                'masked_sentences': sample['input_sentences'],
                'topk': topk,
            })
        
        eval_loss += loss.item() * tot_b
    
    if output_topk is not None:
        logger.info('Output top-k prediction to %s..'%output_topk)
        for rel in list_of_predictions:
            with open(os.path.join(output_topk, '%s.jsonl'%rel), 'w') as f:
                f.write('\n'.join([json.dumps(x) for x in list_of_predictions[rel]]))

    #micro, macro = output_result(result, eval_loss)
    # CHANGE
    # WORKS FOR BINARY
    for rel in list_of_predictions:
        print(os.path.join(output_topk, '%s.jsonl'%rel))
        directory_path = os.path.join(output_topk, '%s.jsonl'%rel)
        jsonl_file = load_jsonl(directory_path)
        # Precaution: works when the whole file have the same target
        target = jsonl_file[0]['obj_label']
        count_target, total = count_binary(jsonl_file, target)


        list_pre_acc.append([template, rel, count_target, total, target])
       
    micro, macro = 0,0
    #return micro, result
    return list_pre_acc


def gen_feature_sample(data_sample, template, obj_data, trusted, mask_token='[MASK]'):
    feature_sample = {}
    feature_sample['predicate_id'] = data_sample['predicate_id']
    feature_sample['sub_label'] = data_sample['sub_label']
    feature_sample['sub_label_2'] = data_sample['sub_label_2']
    feature_sample['obj_label'] = data_sample['obj_label']
    feature_sample['uuid'] = data_sample['uuid'] if 'uuid' in data_sample else ''
    #Change
    #masked_sentence = parse_template(template.strip(), feature_sample['sub_label'].strip(), mask_token)
    masked_sentence = parse_template(template.strip(), feature_sample['sub_label'].strip(), feature_sample['sub_label_2'].strip(), feature_sample['obj_label'], obj_data, trusted, mask_token)
    
    feature_sample['input_sentences'] = [masked_sentence[0]]
    return feature_sample

#CHANGE
def load_data(data_path, template, trusted, vocab_subset=None, mask_token='[MASK]'):
    all_samples = []

    distinct_facts = set()
    raw_samples = load_file(data_path)

    #CHANGE
    
    obj_data = list(set([i['sub_label_2'] for i in raw_samples]))
    for data_sample in raw_samples:
        # follow the LAMA setting, only keep distinct (sub, obj) pairs
        if (data_sample['sub_label'], data_sample['sub_label_2'], data_sample['obj_label']) in distinct_facts:
            continue
        # change
        #if (data_sample['obj_label'] not in vocab_subset):
        #    continue
        distinct_facts.add((data_sample['sub_label'], data_sample['sub_label_2'],  data_sample['obj_label']))


        feature_sample = gen_feature_sample(data_sample, template, obj_data, trusted, mask_token)
        all_samples.append(feature_sample)
    return all_samples


def obtain_results():
    
    list_pre_acc_np = np.array(list_pre_acc)
    count_results = list_pre_acc_np[:,2]
    count_results = count_results.astype(np.float)
    sum_results = np.sum(count_results)

    count_target = list_pre_acc_np[:,3]
    count_target = count_target.astype(np.float)

    # multiplied by two because of binary
    sum_target_ = np.sum(count_target)
    sum_target =  sum_target_ * 2

    acc_binary_half = sum_results/sum_target
    return sum_results, sum_target_, acc_binary_half