import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
load_SFT = False
from FFN_manipulate import *
from attention_manipulate import *
from utils import *
from evaluation import ICL_evaluation, calibration_evaluation
import argparse
import random

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Initial argumentss.')
parser.add_argument('--cuda_device_id',default='0', type=str, help='cuda_device_id')
parser.add_argument('--seed',  type=int, default=10, help='Random seed')
parser.add_argument('--dataset_name',  type=str, default='sst2', help='Dataset Name')
parser.add_argument('--format_index',  type=int, default=None, help='Gen Various Prompt format')
parser.add_argument('--order_index',  type=int, default=None, help='Gen various prompt order')
parser.add_argument('--num_shot',  type=int, default=1, help='Number of shot')
parser.add_argument('--UniBias',  type=bool, default=True, help='Using UniBias or Not')
parser.add_argument('--Calibration',  type=bool, default=True, help='Evaluate Calibration Methods or Not')
args = parser.parse_args()

cuda_device_id = args.cuda_device_id
seed_value = args.seed
random.seed(seed_value)
dataset_name = args.dataset_name
format_index = args.format_index
order_index = args.order_index
num_shot = args.num_shot
Unibias = args.UniBias
Calibration = args.Calibration

os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device_id

# Model setup
device = torch.device("cuda:0")
# change the model path accordingly
model_path = "/mnt/data1/Llama-2-7b-hf"
# model_path = "/mnt/data1/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             )
mlm_head = model.lm_head
norm = model.model.norm
record_file_path = './results/' + dataset_name + '.json'

def main():
    # Load dataset and gen prompts
    # gen prompts with different examples for different random seeds
    if not order_index and not format_index:
        prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_test(dataset_name, num_shot=num_shot)
        validate_data = prepare_dataset_validate(dataset_name, demonstration)
    # gen prompts with different prompt formatting
    if format_index:
        prompt_list, test_labels, demonstration, test_sentences, ans_label_list = gen_test_data_format(dataset_name, format_index)
        validate_data = gen_validate_data_format(dataset_name, demonstration, format_index)
    # gen prompts with different example order
    if order_index:
        prompt_list, test_labels, demonstration, test_sentences, rand_example_sample_index_order = gen_test_data_order(dataset_name, order_index)
        validate_data = gen_validate_data_order(dataset_name, demonstration)

    # labels of the dataset
    ans_label_list = task_labels(dataset_name)
    # find possible token ids for labels
    gt_ans_ids_list = find_possible_ids_for_labels(ans_label_list, tokenizer)

    write_json(record_file_path, dataset_name + ' seed_value: ' + str(seed_value))

    if Unibias:
        # identify and eliminate biased FFN neurons
        biased_FFN_neurons, min_bias_label_logit, debias_alpha_value = biased_FFN_identify_and_eliminate(model, tokenizer, validate_data, ans_label_list, dataset_name)
        write_json(record_file_path, "biased FFN neurons:" + str(biased_FFN_neurons) + str(debias_alpha_value))
        write_json(record_file_path, debias_alpha_value)

        # identify and eliminate biased Attention heads
        biased_AHs, min_bias_label_logit, debias_alpha_value = attention_manipulate(model, tokenizer, validate_data, ans_label_list, dataset_name)
        write_json(record_file_path, "biased attention heads:" + str(biased_AHs) + str(debias_alpha_value))
        write_json(record_file_path, debias_alpha_value)

        # remove common biased components
        # biased_AHs = {"19": [10, 14, 21], "16": [1, 19, 29], "18":[1,31], "25":[21]}
        # debias_alpha = 0
        # set_attention_masks(model, biased_AHs, debias_alpha)

    # evaluate ICL/UniBias performance
    final_acc, all_label_probs, cf = ICL_evaluation(model, prompt_list, test_labels, gt_ans_ids_list, dataset_name)
    if Unibias:
        write_json(record_file_path, 'Unibias: ' + final_acc + str(cf))
    else:
        write_json(record_file_path, final_acc + str(cf))

    # evaluate calibration methods
    if Calibration:
        calibration_evaluation(model, all_label_probs, gt_ans_ids_list, test_sentences, test_labels, demonstration)


if __name__ == "__main__":
    main()