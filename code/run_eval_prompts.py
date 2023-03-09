import argparse
import os
import random
import logging
import torch
import json
import pandas as pd

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, evaluate, get_relation_meta, count_relation_meta, obtain_results
#from acc_qa import count_results

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected (True o False).')


def init_template(prompt_file, relation):
    relation = get_relation_meta(prompt_file, relation)
    return relation['template']

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350M', help='the huggingface model name')
parser.add_argument('--output_dir', type=str, default='output', help='the output directory to store prediction results')
#parser.add_argument('--common_vocab_filename', type=str, default='common_vocab_cased.txt', help='common vocabulary of models (used to filter triples)')
parser.add_argument('--common_vocab_filename', type=str, default='common_vocab_binary.txt', help='common vocabulary of models (used to filter triples)')

parser.add_argument('--prompt_file', type=str, default='prompts/LAMA_relations.jsonl', help='prompt file containing 41 relations')

parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA")
parser.add_argument('--eval_batch_size', type=int, default=32)

parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--output_predictions', default=True, help='whether to output top-k predictions')
parser.add_argument('--k', type=int, default=2, help='how many predictions will be outputted')
parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')
parser.add_argument('--output_all_log_probs', action="store_true", help='whether to output all the log probabilities')
parser.add_argument('--trusted', type=str2bool, nargs='?', const=True, default=True, help='whether to generate trusted or untrusted facts for questions')
parser.add_argument('--filename', type=str, default="example", help = "file name for the output file that holds the results")

if __name__ == "__main__":
    args = parser.parse_args()


    # CHANGE
    trusted = args.trusted
    filename = args.filename
    # Initialize GPUs
    device=torch.device(args.device)
    if args.device == 'cuda':
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            logger.warning('No GPU found! exit!')
        logger.info('# GPUs: %d'%n_gpu)

    elif args.device == 'mps':
        n_gpu = 1
    else:
        logger.info('# Running on CPU')
        n_gpu = 0

    logger.info('Model: %s'%args.model_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)


    print("N GPU:")
    print(n_gpu)
    model = build_model_by_name(args)

    # Turn model.config.output_hidden_states on to get access to hidden states
    model.enable_output_hidden_states()

    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)

    else:
        filter_indices = None
        index_list = None

    if args.output_all_log_probs:
        model.k = len(vocab_subset)

    dict_acc = {}
    FINAL = {}
    for relation in os.listdir(args.test_data_dir):
        relation = relation.split(".")[0]
        print("RELATION {}".format(relation))

        output_dir = os.path.join(args.output_dir, os.path.basename(args.prompt_file).split(".")[0],args.model_name.replace("/","_"))


        os.makedirs(output_dir , exist_ok=True)

        # CHANGE
        count_template, list_templates = count_relation_meta(args.prompt_file, relation)
        print("COUNT TEMPLATE")
        print(count_template)
        #print(list_templates)
        t_count = 0
        final_t = []
        for t in list_templates:
            #if t_count ==2: break
            #t_count +=1
            template = t['template']
        
            #template = init_template(args.prompt_file, relation)
            print("TEMPLATE")
            print(template)


            logger.info('Template: %s'%template)

            test_data = os.path.join(args.test_data_dir, relation + ".jsonl")
                
            #CHANGE
            eval_samples = load_data(test_data, template, trusted, vocab_subset=vocab_subset, mask_token=model.MASK)
    
            eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
    
            list_pre_acc = evaluate(model, eval_samples_batches, eval_sentences_batches, template, filter_indices, index_list, output_topk=output_dir if args.output_predictions else None)
    
            d_path = 'results_acc/'
            if not os.path.exists(d_path):
                os.makedirs(d_path)

            final_t.append(list_pre_acc)
            #print(final_t)
            
        FINAL[relation] = final_t
        #print(FINAL)
    
    #print("FINAL variable with all the results accumulated")
    print(FINAL)
    
    # Directory where filtered files will be
    directory_path_filtered = 'results/'
    if not os.path.exists(directory_path_filtered):
        os.makedirs(directory_path_filtered)

    path_files = os.path.join(directory_path_filtered, str(filename)+'.txt')

    with open(path_files, 'w') as convert_file:
     convert_file.write(json.dumps(FINAL))
