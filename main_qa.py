import argparse
from making_prompt_jsonl import making_prompt_jsonl
from build_demonstration_support import making_multiple_prompt_jsonl
from filtered_LAMA_QA import filtered_LAMA_QA
import argparse
import os
import random
import logging

'''
Example to call this file: 
python main_qa.py --trusted True --target "yes" --question "The answer is"
'''

def main(args):
    resultado = suma(args.a, args.b)
    #print(f"El resultado de la suma es: {resultado}")



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected (True o False).')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Suma dos valores enteros')
    parser.add_argument('--trusted', type=str2bool, nargs='?', const=True, default=True, help='whether to generate trusted or untrusted facts for questions')

    parser.add_argument('--target', type=str, help='The target declared in filtered LAMA. Ex: yes, no, Yes, No, etc.')
    parser.add_argument('--question', type=str, help='Extra words added to prompt, example: The answer is')
    parser.add_argument('--sequences', type=str, default="", help='Extra sequences before the fact to be evaluated')
    parser.add_argument('--templates', type=str, default="", help='Whether to use same template for extra sequences or not. Options: normal, random')


    args = parser.parse_args()

    trusted = args.trusted
    target = args.target
    question = args.question
    sequences = args.sequences
    templates = args.templates
    # making filtered data
    filtered_LAMA_QA(target)

    if len(templates) > 0 and len(sequences) > 0:
        making_multiple_prompt_jsonl(trusted, target, question, sequences, templates)

    # making prompt/LAMA_relations.jsonl based on trusted and question values
    else:

        making_prompt_jsonl(trusted, question)
    

