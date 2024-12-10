import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import transformers
import torch
import re
from datasets import load_dataset
import random
import numpy as np
from essential_generators import DocumentGenerator
import itertools
import ast
import argparse

def prepare_dataset_test(dataset_name, content_free_inputs=None, demonstration=None, num_shot=1):
    '''gen testing prompts: slecting different examples for different random seeds'''
    if dataset_name == 'sst2':
        prompt_list = []
        dataset_test = load_dataset("sst2", split="validation")
        test_samples = dataset_test['sentence']
        test_labels = dataset_test['label']
        test_sentences = test_samples
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration
            examples = load_dataset("sst2", split="train")
            examples_samples = examples['sentence']
            examples_labels = examples['label']
            examples_answers = ["positive" if i == 1 else 'negative' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            example_pos_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_neg_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            rand_example_pos_index = random.sample(example_pos_index, num_shot)
            rand_example_neg_index = random.sample(example_neg_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_pos_index, rand_example_neg_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Classify the sentiment of the review. positive or negative?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Sentiment: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'sst5':
        prompt_list = []
        dataset_test = load_dataset("SetFit/sst5", split="test")
        test_samples = dataset_test['text']
        test_labels = dataset_test['label']
        test_sentences = test_samples
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration
            examples = load_dataset("SetFit/sst5", split="train")
            examples_samples = examples['text']
            examples_labels = examples['label']
            examples_answers = ["terrible" if i == 0 else 'bad' if i == 1 else 'okay' if i == 2 else 'good' if i == 3 else 'great' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            example_indices = {label: [] for label in range(len(set(examples_labels)))}
            for i, label in enumerate(examples_labels):
                if label in example_indices:
                    example_indices[label].append(i)
            random_sample_indices = []
            for label in range(len(example_indices)):
                sampled_indices = random.sample(example_indices[label], num_shot)
                random_sample_indices.append(sampled_indices)
            rand_example_sample_index = [elem for pair in zip(*random_sample_indices) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Classify the sentiment of the review. terrible, bad, okay, good or great?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Sentiment: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'trec':
        prompt_list = []
        dataset_test = load_dataset(dataset_name, split="test")
        test_samples = dataset_test['text']
        test_labels = dataset_test['coarse_label']
        test_sentences = test_samples
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Question: ' + sample + '\n' + 'Answer Type:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration
            examples = load_dataset(dataset_name, split="train")
            examples_samples = examples['text']
            examples_labels = examples['coarse_label']
            examples_answers = ["abbreviation" if i == 0 else 'entity' if i == 1 else 'description' if i == 2 else 'person' if i == 3 else 'location' if i == 4 else 'number' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            example_indices = {label: [] for label in range(len(set(examples_labels)))}
            for i, label in enumerate(examples_labels):
                if label in example_indices:
                    example_indices[label].append(i)
            random_sample_indices = []
            for label in range(len(example_indices)):
                sampled_indices = random.sample(example_indices[label], num_shot)
                random_sample_indices.append(sampled_indices)
            rand_example_sample_index = [elem for pair in zip(*random_sample_indices) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Classify the type of the answer to the question. abbreviation, entity, description, person, location or number?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Question: ' + example_pairs[rand_index][0] + '\n' + 'Answer Type: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Question: ' + sample + '\n' + 'Answer Type:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'mnli':
        prompt_list = []
        dataset_test = load_dataset("nyu-mll/glue", dataset_name, split="validation_matched")[:3000]
        examples = load_dataset("nyu-mll/glue", dataset_name, split="train")
        test_premise = dataset_test['premise']
        test_hypothesis = dataset_test['hypothesis']
        test_labels = dataset_test['label']
        test_sentences = list([i,j] for i, j in zip(test_premise, test_hypothesis))
        if content_free_inputs:
            # instruction = 'Based on the premise,can we conclude the hypothesis is true? Yes, no, or maybe?'
            instruction = ''
            for sample in content_free_inputs:
                prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + '\n' + instruction + '\n' + 'Answer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            if dataset_name == 'sick':
                examples_premise = examples['sentence_A']
                examples_hypothesis = examples['sentence_B']
            else:
                examples_premise = examples['premise']
                examples_hypothesis = examples['hypothesis']
            examples_labels = examples['label']
            examples_answers = ["yes" if i == 0 else 'maybe' if i == 1 else 'no' for i in examples_labels]
            example_pairs = list(zip(examples_premise, examples_hypothesis, examples_answers))
            example_yes_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_may_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_no_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 2]
            rand_example_ent_index = random.sample(example_yes_index, num_shot)
            rand_example_neu_index = random.sample(example_may_index, num_shot)
            rand_example_con_index = random.sample(example_no_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_ent_index, rand_example_neu_index, rand_example_con_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction='Given the premise, are we justified in saying that hypothesis? yes, no, or maybe?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i =  'Premise: ' + example_pairs[rand_index][0] + '\nHypothesis: ' + example_pairs[rand_index][1] + instruction + '\nAnswer: ' + example_pairs[rand_index][2] + '\n\n'
                demonstration += example_i
            for premise, hypothesis in zip(test_premise, test_hypothesis):
                prompt = demonstration + 'Premise: ' + premise + '\nHypothesis: ' + hypothesis + instruction + '\n' + 'Answer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'ag_news':
        prompt_list = []
        dataset_test = load_dataset(dataset_name, split="test")[:3000]
        test_text = dataset_test['text']
        test_labels = dataset_test['label']
        test_sentences = test_text
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Article: ' + sample + '\n' + 'Answer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset(dataset_name, split="train")
            examples_text = examples['text']
            examples_labels = examples['label']
            examples_answers = ["world" if i == 0 else 'sports' if i==1 else 'business' if i == 2 else 'technology & science' for i in examples_labels]
            # examples_answers = ["world" if i == 0 else 'sports' if i==1 else 'business' if i == 2 else 'technology' for i in examples_labels]
            example_pairs = list(zip(examples_text, examples_answers))
            example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_2_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 2]
            example_3_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 3]
            rand_example_0_index = random.sample(example_0_index, num_shot)
            rand_example_1_index = random.sample(example_1_index, num_shot)
            rand_example_2_index = random.sample(example_2_index, num_shot)
            rand_example_3_index = random.sample(example_3_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_0_index, rand_example_1_index, rand_example_2_index, rand_example_3_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Classify the news articles into the categories of world, sports, business, and technology & science.\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Article: ' + example_pairs[rand_index][0] + '\n' + 'Answer: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            # demonstration = ''
            for text in test_text:
                prompt = demonstration + 'Article: ' + text + '\n' + 'Answer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name in ('cr', 'mr'):
        prompt_list = []
        if dataset_name == 'cr':
            dataset_test = load_dataset("SetFit/CR", split="test")
            examples = load_dataset("SetFit/CR", split="train")
        elif dataset_name == 'mr':
            dataset_test = load_dataset("mattymchen/mr", split="test")[:3000]
            examples = load_dataset("mattymchen/mr", split="test")[3000:]
        test_samples = dataset_test['text']
        test_labels = dataset_test['label']
        test_sentences = test_samples
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples_samples = examples['text']
            examples_labels = examples['label']
            examples_answers = ["positive" if i == 1 else 'negative' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            example_pos_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_neg_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            rand_example_pos_index = random.sample(example_pos_index, num_shot)
            rand_example_neg_index = random.sample(example_neg_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_pos_index, rand_example_neg_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Classify the sentiment of the review. positive or negative?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Sentiment: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'copa':
        prompt_list = []
        dataset_test = load_dataset("super_glue", dataset_name, split="validation")
        test_premise = dataset_test['premise']
        test_c1 = dataset_test['choice1']
        test_c2 = dataset_test['choice2']
        test_samples = list(zip(test_premise, test_c1, test_c2))
        test_labels = dataset_test['label']
        test_sentences = list([i,j,k] for i,j,k in zip(test_premise, test_c1, test_c2))
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Premise: ' + sample[0] + '\nChoice 1: ' + sample[1] + '\nChoice 2: ' + sample[2] + '\n' + 'Answer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset("super_glue", dataset_name, split="train")
            examples_premise = examples['premise']
            examples_c1 = examples['choice1']
            examples_c2 = examples['choice2']
            examples_labels = examples['label']
            examples_answers = ["2" if i == 1 else '1' for i in examples_labels]
            example_pairs = list(zip(examples_premise, examples_c1, examples_c2, examples_answers))
            example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            rand_example_1_index = random.sample(example_1_index, num_shot)
            rand_example_0_index = random.sample(example_0_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_1_index, rand_example_0_index) for elem in pair]
            demonstration = ''
            for rand_index in rand_example_sample_index:
                example_i = 'Premise: ' + example_pairs[rand_index][0] + '\nChoice 1: ' + example_pairs[rand_index][1] + '\nChoice 2: ' + example_pairs[rand_index][2] + '\nAnswer: ' + example_pairs[rand_index][3] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Premise: ' + sample[0] + '\nChoice 1: ' + sample[1] + '\nChoice 2: ' + sample[2] + '\n' + 'Answer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'rte':
        prompt_list = []
        dataset_test = load_dataset("super_glue", dataset_name, split="validation")
        test_premise = dataset_test['premise']
        test_hypothesis = dataset_test['hypothesis']
        test_labels = dataset_test['label']
        test_sentences = list([i,j] for i,j in zip(test_premise, test_hypothesis))
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + '\n' + 'Answer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset("super_glue", dataset_name, split="train")
            examples_premise = examples['premise']
            examples_hypothesis = examples['hypothesis']
            examples_labels = examples['label']
            examples_answers = ["yes" if i == 0 else 'no' for i in examples_labels]
            example_pairs = list(zip(examples_premise, examples_hypothesis, examples_answers))
            example_yes_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_no_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            rand_example_ent_index = random.sample(example_yes_index, num_shot)
            rand_example_con_index = random.sample(example_no_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_ent_index, rand_example_con_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = '\nGiven the premise, are we justified in saying that hypothesis? yes or no?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Premise: ' + example_pairs[rand_index][0] + '\nHypothesis: ' + example_pairs[rand_index][1] + '\nAnswer: ' + example_pairs[rand_index][2] + '\n\n'
                demonstration += example_i
            for premise, hypothesis in zip(test_premise, test_hypothesis):
                prompt = demonstration + 'Premise: ' + premise + '\nHypothesis: ' + hypothesis + '\n' + 'Answer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'wic':
        prompt_list = []
        dataset_test = load_dataset("super_glue", dataset_name, split="validation")
        test_s1 = dataset_test['sentence1']
        test_s2 = dataset_test['sentence2']
        test_word = dataset_test['word']
        test_samples = list(zip(test_s1, test_s2, test_word))
        test_labels = dataset_test['label']
        test_sentences = list([i,j,k] for i,j,k in zip(test_s1, test_s2, test_word))
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Sentence 1: ' + sample[0] + '\nSentence 2: ' + sample[1] +'\nWord: ' + sample[2] + '\n' + 'Answer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset("super_glue", dataset_name, split="train")
            examples_s1 = examples['sentence1']
            examples_s2 = examples['sentence2']
            examples_word = examples['word']
            examples_labels = examples['label']
            examples_answers = ["false" if i == 0 else 'true' for i in examples_labels]
            example_pairs = list(zip(examples_s1, examples_s2, examples_word, examples_answers))
            example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            rand_example_0_index = random.sample(example_0_index, num_shot)
            rand_example_1_index = random.sample(example_1_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_0_index, rand_example_1_index) for elem in pair]
            demonstration = ''
            if num_shot == 0:
                instruction = 'Identify if the word used in the same way in the two sentences below. true or false?\n\n'
            else:
                instruction = ''
            demonstration += instruction
            for rand_index in rand_example_sample_index:
                example_i = 'Sentence 1: ' + example_pairs[rand_index][0] + '\nSentence 2: ' + example_pairs[rand_index][1] + '\nWord: ' + example_pairs[rand_index][2] + '\nAnswer: ' + example_pairs[rand_index][3] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Sentence 1: ' + sample[0] + '\nSentence 2: ' + sample[1] +'\nWord: ' + sample[2] + '\n' + 'Answer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'arc':
        prompt_list = []
        dataset_test = load_dataset('ai2_arc', 'ARC-Challenge', split="test")
        test_question = dataset_test['question']
        test_choices = dataset_test['choices']
        test_choices_list = [[f"{label}. {text}" for label, text in zip(sample_i['label'], sample_i['text'])] for sample_i in test_choices]
        test_keys = dataset_test['answerKey']
        key_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4':3}
        test_labels = [key_to_int[key] for key in test_keys]
        test_sentences = list([i]+j for i, j in zip(test_question, test_choices_list))
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset('ai2_arc', 'ARC-Challenge', split="train")
            examples_question = examples['question']
            examples_choices = examples['choices']
            examples_choices_list = [[f"{label}. {text}" for label, text in zip(sample_i['label'], sample_i['text'])] for sample_i in examples_choices]
            examples_keys = examples['answerKey']
            examples_labels = [key_to_int[key] for key in examples_keys]
            example_pairs = list([i] + j + [k] for i, j, k in zip(examples_question, examples_choices_list, examples_keys))
            example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_2_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 2]
            example_3_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 3]
            rand_example_0_index = random.sample(example_0_index, num_shot)
            rand_example_1_index = random.sample(example_1_index, num_shot)
            rand_example_2_index = random.sample(example_2_index, num_shot)
            rand_example_3_index = random.sample(example_3_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_0_index, rand_example_1_index, rand_example_2_index, rand_example_3_index) for elem in pair]
            demonstration = ''
            for rand_index in rand_example_sample_index:
                example_i = 'Question: ' +  '\n'.join([str(elem) for elem in example_pairs[rand_index][:-1]]) + '\nAnswer: ' + example_pairs[rand_index][-1] + '\n\n'
                demonstration += example_i
            demonstration = ''
            for test_sample_i in test_sentences:
                prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in test_sample_i]) + '\nAnswer:'
                # prompt = demonstration + 'Question: ' + '\n'.join([str(test_sample_i[0])] + [str(elem) for elem in test_sample_i[2:]] + [str(test_sample_i[1])]) + '\nAnswer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences
    if dataset_name == 'mmlu':
        prompt_list = []
        dataset_test = load_dataset("cais/mmlu", 'all', split="test").shuffle()[:2000]
        test_question = dataset_test['question']
        test_choices = dataset_test['choices']
        test_choices_list = [[f"{label}. {text}" for label, text in zip(['A','B', 'C', 'D'], sample_i)] for sample_i in test_choices]
        test_labels = dataset_test['answer']
        test_sentences = list([i]+j for i, j in zip(test_question, test_choices_list))
        if content_free_inputs:
            for sample in content_free_inputs:
                prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer:'
                prompt_list.append(prompt)
            test_labels = None
        else:
            # generate demonstration examples
            examples = load_dataset("cais/mmlu", 'all', split="validation")
            examples_question = examples['question']
            examples_choices = examples['choices']
            examples_choices_list = [[f"{label}. {text}" for label, text in zip(['A','B', 'C', 'D'], sample_i)] for sample_i in examples_choices]
            examples_labels = examples['answer']
            label_to_key = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            examples_keys = [label_to_key[label] for label in examples_labels]
            example_pairs = list([i] + j + [k] for i, j, k in zip(examples_question, examples_choices_list, examples_keys))
            example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
            example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
            example_2_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 2]
            example_3_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 3]
            rand_example_0_index = random.sample(example_0_index, num_shot)
            rand_example_1_index = random.sample(example_1_index, num_shot)
            rand_example_2_index = random.sample(example_2_index, num_shot)
            rand_example_3_index = random.sample(example_3_index, num_shot)
            rand_example_sample_index = [elem for pair in zip(rand_example_0_index, rand_example_1_index, rand_example_2_index, rand_example_3_index) for elem in pair]
            demonstration = ''
            for rand_index in rand_example_sample_index:
                example_i = 'Question: ' +  '\n'.join([str(elem) for elem in example_pairs[rand_index][:-1]]) + '\nAnswer: ' + example_pairs[rand_index][-1] + '\n\n'
                demonstration += example_i
            for test_sample_i in test_sentences:
                prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in test_sample_i]) + '\nAnswer:'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences


def prepare_dataset_validate(dataset_name, demonstration=None):
    '''Generate support set for grid seearch'''
    if dataset_name == 'sst2':
        prompt_list = []
        dataset_validate = load_dataset("sst2", split="train")
        valid_samples = dataset_validate['sentence']
        valid_labels = dataset_validate['label']
        valid_pos_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_neg_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        random_pos_index = random.sample(valid_pos_index, 20)
        random_neg_index = random.sample(valid_neg_index, 20)
        random_sample_index = sorted(random_pos_index + random_neg_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        for sample in sampled_valid_samples:
            random_label = random.sample(['negative', 'positive'],1)[0]
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'sst5':
        prompt_list = []
        dataset_validate = load_dataset("SetFit/sst5", split="train")
        valid_samples = dataset_validate['text']
        valid_labels = dataset_validate['label']
        valid_indices = {label: [] for label in range(len(set(valid_labels)))}
        for i, label in enumerate(valid_labels):
            if label in valid_indices:
                valid_indices[label].append(i)
        random_sample_indices = []
        for label in range(len(valid_indices)):
            sampled_indices = random.sample(valid_indices[label], 20)
            random_sample_indices.extend(sampled_indices)
        random_sample_indices.sort()
        sampled_valid_samples = [valid_samples[i] for i in random_sample_indices]
        for sample in sampled_valid_samples:
            random_label = random.sample(['terrible', 'bad', 'okay', 'good', 'great'],1)[0]
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'mnli':
        prompt_list = []
        dataset_validate = load_dataset("nyu-mll/glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_hypothesis = dataset_validate['hypothesis']
        valid_labels = dataset_validate['label']
        valid_samples = [(i,j) for i, j in zip(valid_premise, valid_hypothesis)]
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_2_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 2]
        random_0_index = random.sample(valid_0_index, 20)
        random_1_index = random.sample(valid_1_index, 20)
        random_2_index = random.sample(valid_2_index, 20)
        random_sample_index = sorted(random_0_index + random_1_index + random_2_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['yes', 'maybe', 'no'],1)[0]
            # instruction = '\nGiven the premise, are we justified in saying that hypothesis? Yes, no, or maybe?'
            instruction = ''
            prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + instruction + '\nAnswer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'ag_news':
        prompt_list = []
        dataset_validate = load_dataset(dataset_name, split="train")
        valid_text = dataset_validate['text']
        valid_labels = dataset_validate['label']
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_2_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 2]
        valid_3_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 3]
        random_0_index = random.sample(valid_0_index, 20)
        random_1_index = random.sample(valid_1_index, 20)
        random_2_index = random.sample(valid_2_index, 20)
        random_3_index = random.sample(valid_3_index, 20)
        random_sample_index = sorted(random_0_index + random_1_index + random_2_index + random_3_index)
        sampled_valid_samples = [valid_text[i] for i in random_sample_index]
        for sample in sampled_valid_samples:
            random_label = random.sample(['world', 'sports', 'business', 'technology & science'],1)[0]
            # random_label = random.sample(['world', 'sports', 'business', 'technology'], 1)[0]
            prompt = demonstration + 'Article: ' + sample + '\n' + 'Answer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name in ('cr', 'mr'):
        prompt_list = []
        if dataset_name == 'cr':
            dataset_validate = load_dataset("SetFit/CR", split="train")
        elif dataset_name == 'mr':
            dataset_validate = load_dataset("mattymchen/mr", split="test")[3000:]
        valid_samples = dataset_validate['text']
        valid_labels = dataset_validate['label']
        valid_pos_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_neg_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        random_pos_index = random.sample(valid_pos_index, 20)
        random_neg_index = random.sample(valid_neg_index, 20)
        random_sample_index = sorted(random_pos_index + random_neg_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        for sample in sampled_valid_samples:
            random_label = random.sample(['positive', 'negative'], 1)[0]
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'copa':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_c1 = dataset_validate['choice1']
        valid_c2 = dataset_validate['choice2']
        valid_samples = list(zip(valid_premise, valid_c1, valid_c2))
        valid_labels = dataset_validate['label']
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        random_1_index = random.sample(valid_1_index, 20)
        random_0_index = random.sample(valid_0_index, 20)
        random_sample_index = sorted(random_1_index + random_0_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        for sample in sampled_valid_samples:
            random_label = random.sample(['1', '2'], 1)[0]
            prompt = demonstration + 'Premise: ' + sample[0] + '\nChoice 1: ' + sample[1] + '\nChoice 2: ' + sample[2] + '\n' + 'Answer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'rte':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_hypothesis = dataset_validate['hypothesis']
        valid_labels = dataset_validate['label']
        valid_samples = list(zip(valid_premise, valid_hypothesis))
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        random_0_index = random.sample(valid_0_index, 20)
        random_1_index = random.sample(valid_1_index, 20)
        random_sample_index = sorted(random_0_index + random_1_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['yes', 'no'],1)[0]
            prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + '\nAnswer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'wic':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_s1 = dataset_validate['sentence1']
        valid_s2 = dataset_validate['sentence2']
        valid_word = dataset_validate['word']
        valid_labels = dataset_validate['label']
        valid_samples = list(zip(valid_s1, valid_s2, valid_word))
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        random_0_index = random.sample(valid_0_index, 20)
        random_1_index = random.sample(valid_1_index, 20)
        random_sample_index = sorted(random_0_index + random_1_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['false', 'true'],1)[0]
            prompt = demonstration + 'Sentence 1: ' + sample[0] + '\nSentence 2: ' + sample[1] + '\nWord: ' + sample[2] + '\nAnswer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'arc':
        prompt_list = []
        dataset_validate = load_dataset('ai2_arc', 'ARC-Challenge', split="train")
        valid_question = dataset_validate['question']
        valid_choices = dataset_validate['choices']
        valid_choices_list = [[f"{label}. {text}" for label, text in zip(sample_i['label'], sample_i['text'])] for sample_i in valid_choices]
        valid_keys = dataset_validate['answerKey']
        key_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4':3}
        valid_labels = [key_to_int[key] for key in valid_keys]
        valid_samples = list([i] + j for i, j in zip(valid_question, valid_choices_list))
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_2_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 2]
        valid_3_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 3]
        rand_0_index = random.sample(valid_0_index, 15)
        rand_1_index = random.sample(valid_1_index, 15)
        rand_2_index = random.sample(valid_2_index, 15)
        rand_3_index = random.sample(valid_3_index, 15)
        random_sample_index = sorted(rand_0_index + rand_1_index + rand_2_index + rand_3_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        # sampled_valid_labels = [valid_labels[i] for i in random_sample_index]
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['A', 'B', 'C', 'D'], 1)[0]
            prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'mmlu':
        prompt_list = []
        dataset_validate = load_dataset("cais/mmlu", 'all', split="validation")
        valid_question = dataset_validate['question']
        valid_choices = dataset_validate['choices']
        valid_choices_list = [[f"{label}. {text}" for label, text in zip(['A','B', 'C', 'D'], sample_i)] for sample_i in valid_choices]
        valid_labels = dataset_validate['answer']
        valid_samples = list([i] + j for i, j in zip(valid_question, valid_choices_list))
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_2_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 2]
        valid_3_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 3]
        rand_0_index = random.sample(valid_0_index, 20)
        rand_1_index = random.sample(valid_1_index, 20)
        rand_2_index = random.sample(valid_2_index, 20)
        rand_3_index = random.sample(valid_3_index, 20)
        random_sample_index = sorted(rand_0_index + rand_1_index + rand_2_index + rand_3_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['A', 'B', 'C', 'D'], 1)[0]
            prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list

def task_labels(dataset_name):
    if dataset_name == 'cr':
        ans_token_list = [['negative'], ['positive']]
    if dataset_name == 'wic':
        ans_token_list = [['false'], ['true']]
    if dataset_name == 'copa':
        ans_token_list = [['1'], ['2']]
    if dataset_name == 'rte':
        ans_token_list = [['yes'], ['no']]
    if dataset_name == 'arc':
        ans_token_list = [['A', 'a'], ['B', 'b'], ['C', 'c'], ['D', 'd']]
    if dataset_name == 'mmlu':
        ans_token_list = [['A', 'a'], ['B', 'b'], ['C', 'c'], ['D', 'd']]
    if dataset_name == 'sst2':
        ans_token_list = [['negative'], ['positive']]
    if dataset_name == 'mr':
        ans_token_list = [['negative'], ['positive']]
    if dataset_name == 'sst5':
        ans_token_list = [['terrible'], ['bad'], ['okay'], ['good'], ['great']]
    if dataset_name == 'mnli':
        ans_token_list = [['yes'], ['maybe'], ['no']]
    if dataset_name == 'trec':
        ans_token_list = [['abbreviation'], ['entity'], ['description'], ['person'], ['location'], ['number']]
    if dataset_name == 'ag_news':
        ans_token_list = [['world'], ['sports'], ['business'], ['technology', 'science']]
    return ans_token_list

def gen_PC_calibration_data(dataset_name, demonstration=None):
    if dataset_name == 'sst2':
        prompt_list = []
        dataset_validate = load_dataset("sst2", split="train")
        valid_samples = dataset_validate['sentence']
        valid_labels = dataset_validate['label']
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        for sample in sampled_valid_samples:
            random_label = random.sample(['negative', 'positive'],1)[0]
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'sst5':
        prompt_list = []
        dataset_validate = load_dataset("SetFit/sst5", split="train")
        valid_samples = dataset_validate['text']
        valid_labels = dataset_validate['label']
        data_num = 100 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'trec':
        prompt_list = []
        dataset_validate = load_dataset(dataset_name, split="train")
        valid_samples = dataset_validate['text']
        valid_labels = dataset_validate['coarse_label']
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Question: ' + sample + '\n' + 'Answer Type:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'mnli':
        prompt_list = []
        dataset_validate = load_dataset("nyu-mll/glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_hypothesis = dataset_validate['hypothesis']
        valid_labels = dataset_validate['label']
        valid_samples = [(i,j) for i, j in zip(valid_premise, valid_hypothesis)]
        data_num = min(200 * len(set(valid_labels)), len(valid_samples))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            random_label = random.sample(['yes', 'maybe', 'no'],1)[0]
            # instruction = '\nGiven the premise, are we justified in saying that hypothesis? Yes, no, or maybe?'
            instruction = ''
            prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + instruction + '\nAnswer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'ag_news':
        prompt_list = []
        dataset_validate = load_dataset(dataset_name, split="train")
        valid_text = dataset_validate['text']
        valid_labels = dataset_validate['label']
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_text, data_num)
        for sample in sampled_valid_samples:
            random_label = random.sample(['world', 'sports', 'business', 'technology & science'],1)[0]
            prompt = demonstration + 'Article: ' + sample + '\n' + 'Answer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name in ('cr', 'mr'):
        prompt_list = []
        if dataset_name == 'cr':
            dataset_validate = load_dataset("SetFit/CR", split="train")
        elif dataset_name == 'mr':
            dataset_validate = load_dataset("mattymchen/mr", split="test")[3000:]
        valid_samples = dataset_validate['text']
        valid_labels = dataset_validate['label']
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        for sample in sampled_valid_samples:
            random_label = random.sample(['positive', 'negative'], 1)[0]
            prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'copa':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_c1 = dataset_validate['choice1']
        valid_c2 = dataset_validate['choice2']
        valid_samples = list(zip(valid_premise, valid_c1, valid_c2))
        valid_labels = dataset_validate['label']
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Premise: ' + sample[0] + '\nChoice 1: ' + sample[1] + '\nChoice 2: ' + sample[2] + '\n' + 'Answer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'rte':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_premise = dataset_validate['premise']
        valid_hypothesis = dataset_validate['hypothesis']
        valid_labels = dataset_validate['label']
        valid_samples = list(zip(valid_premise, valid_hypothesis))
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Premise: ' + sample[0] + '\nHypothesis: ' + sample[1] + '\nAnswer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'wic':
        prompt_list = []
        dataset_validate = load_dataset("super_glue", dataset_name, split="train")
        valid_s1 = dataset_validate['sentence1']
        valid_s2 = dataset_validate['sentence2']
        valid_word = dataset_validate['word']
        valid_labels = dataset_validate['label']
        valid_samples = list(zip(valid_s1, valid_s2, valid_word))
        data_num = 200 * len(set(valid_labels))
        sampled_valid_samples = random.sample(valid_samples, data_num)
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Sentence 1: ' + sample[0] + '\nSentence 2: ' + sample[1] + '\nWord: ' + sample[2] + '\nAnswer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'arc':
        prompt_list = []
        dataset_validate = load_dataset('ai2_arc', 'ARC-Challenge', split="train")
        valid_question = dataset_validate['question']
        valid_choices = dataset_validate['choices']
        valid_choices_list = [[f"{label}. {text}" for label, text in zip(sample_i['label'], sample_i['text'])] for sample_i in valid_choices]
        valid_keys = dataset_validate['answerKey']
        key_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4':3}
        valid_labels = [key_to_int[key] for key in valid_keys]
        valid_samples = list([i] + j for i, j in zip(valid_question, valid_choices_list))
        data_num = 200 * 4
        sampled_valid_samples = random.sample(valid_samples, data_num)
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer:'
            prompt_list.append(prompt)
        return prompt_list
    if dataset_name == 'mmlu':
        prompt_list = []
        dataset_validate = load_dataset("cais/mmlu", 'all', split="validation")
        valid_question = dataset_validate['question']
        valid_choices = dataset_validate['choices']
        valid_choices_list = [[f"{label}. {text}" for label, text in zip(['A','B', 'C', 'D'], sample_i)] for sample_i in valid_choices]
        valid_labels = dataset_validate['answer']
        valid_samples = list([i] + j for i, j in zip(valid_question, valid_choices_list))
        data_num = 100 * 4
        sampled_valid_samples = random.sample(valid_samples, data_num)
        demonstration = demonstration
        # generate validate data with random labels
        for sample in sampled_valid_samples:
            prompt = demonstration + 'Question: ' + '\n'.join([str(elem) for elem in sample]) + '\nAnswer:'
            prompt_list.append(prompt)
        return prompt_list

def gen_test_data_format(dataset_name, format_index = None):
    if dataset_name == 'sst2':
        prompt_list = []
        dataset_test = load_dataset("sst2", split="validation")
        test_samples = dataset_test['sentence']
        test_labels = dataset_test['label']
        test_sentences = test_samples
        # generate demonstration
        examples = load_dataset("sst2", split="train")
        examples_samples = examples['sentence']
        examples_labels = examples['label']
        example_pos_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
        example_neg_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
        rand_example_pos_index = random.sample(example_pos_index, 1)
        rand_example_neg_index = random.sample(example_neg_index, 1)
        rand_example_sample_index = [elem for pair in zip(rand_example_pos_index, rand_example_neg_index) for elem in pair]
        # rand_example_sample_index = [elem for pair in zip(rand_example_neg_index, rand_example_pos_index) for elem in pair]
        demonstration = ''
        if format_index == 0:
            ans_token_list = [['negative'], ['positive']]
            examples_answers = ["positive" if i == 1 else 'negative' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Sentiment: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
        if format_index == 1:
            ans_token_list = [['negative'], ['positive']]
            examples_answers = ["positive" if i == 1 else 'negative' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Input: ' + example_pairs[rand_index][0] + '\n' + 'Prediction: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Input: ' + sample + '\n' + 'Prediction:'
                prompt_list.append(prompt)
        if format_index == 2:
            ans_token_list = [['bad'], ['good']]
            examples_answers = ["good" if i == 1 else 'bad' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Sentiment: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment:'
                prompt_list.append(prompt)
        if format_index == 3:
            ans_token_list = [['bad'], ['good']]
            examples_answers = ["good" if i == 1 else 'bad' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = example_pairs[rand_index][0] + ' It was ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + sample + ' It was'
                prompt_list.append(prompt)
        if format_index == 4:
            ans_token_list = [['no'], ['yes']]
            examples_answers = ["yes" if i == 1 else 'no' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Positive Review: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Positive Review:'
                prompt_list.append(prompt)
        if format_index == 5:
            ans_token_list = [['0'], ['5']]
            examples_answers = ["5" if i == 1 else '0' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Stars: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Stars:'
                prompt_list.append(prompt)
        if format_index == 6:
            ans_token_list = [['bad'], ['good']]
            examples_answers = ["good" if i == 1 else 'bad' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = example_pairs[rand_index][0] + ' My overall feeling was that the movie was ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + sample + ' My overall feeling was that the movie was'
                prompt_list.append(prompt)
        if format_index == 7:
            ans_token_list = [['negative'], ['positive']]
            examples_answers = ["positive" if i == 1 else 'negative' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'Review: ' + example_pairs[rand_index][0] + '\n' + 'Question: Is the sentiment of the above review Positive or Negative?\n' + 'Answer: ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Question: Is the sentiment of the above review Positive or Negative?\n' + 'Answer:'
                prompt_list.append(prompt)
        if format_index == 8:
            ans_token_list = [['bad'], ['good']]
            examples_answers = ["good" if i == 1 else 'bad' for i in examples_labels]
            example_pairs = list(zip(examples_samples, examples_answers))
            for rand_index in rand_example_sample_index:
                example_i = 'My review for last night\'s ﬁlm: ' + example_pairs[rand_index][0] + ' The critics agreed that this movie was ' + example_pairs[rand_index][1] + '\n\n'
                demonstration += example_i
            for sample in test_samples:
                prompt = demonstration + 'My review for last night\'s ﬁlm: ' + sample +  ' The critics agreed that this movie was'
                prompt_list.append(prompt)
        return prompt_list, test_labels, demonstration, test_sentences, ans_token_list
    else:
        print('only provide different prompt formats for sst2')

def gen_validate_data_format(dataset_name, demonstration=None, format_index = None):
    if dataset_name == 'sst2':
        prompt_list = []
        dataset_validate = load_dataset("sst2", split="train")
        valid_samples = dataset_validate['sentence']
        valid_labels = dataset_validate['label']
        valid_pos_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_neg_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        random_pos_index = random.sample(valid_pos_index, 20)
        random_neg_index = random.sample(valid_neg_index, 20)
        random_sample_index = sorted(random_pos_index + random_neg_index)
        sampled_valid_samples = [valid_samples[i] for i in random_sample_index]
        if format_index == 0:
            for sample in sampled_valid_samples:
                random_label = random.sample(['negative', 'positive'], 1)[0]
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment: ' + random_label
                prompt_list.append(prompt)
        if format_index == 1:
            for sample in sampled_valid_samples:
                random_label = random.sample(['negative', 'positive'], 1)[0]
                prompt = demonstration + 'Input: ' + sample + '\n' + 'Prediction: ' + random_label
                prompt_list.append(prompt)
        if format_index == 2:
            for sample in sampled_valid_samples:
                random_label = random.sample(['good', 'bad'], 1)[0]
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Sentiment: ' + random_label
                prompt_list.append(prompt)
        if format_index == 3:
            for sample in sampled_valid_samples:
                random_label = random.sample(['good', 'bad'], 1)[0]
                prompt = demonstration + sample + ' It was ' + random_label
                prompt_list.append(prompt)
        if format_index == 4:
            for sample in sampled_valid_samples:
                random_label = random.sample(['yes', 'no'], 1)[0]
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Positive Review: ' + random_label
                prompt_list.append(prompt)
        if format_index == 5:
            for sample in sampled_valid_samples:
                random_label = random.sample(['5', '0'], 1)[0]
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Stars: ' + random_label
                prompt_list.append(prompt)
        if format_index == 6:
            for sample in sampled_valid_samples:
                random_label = random.sample(['good', 'bad'], 1)[0]
                prompt = demonstration + sample + ' My overall feeling was that the movie was ' + random_label
                prompt_list.append(prompt)
        if format_index == 7:
            for sample in sampled_valid_samples:
                random_label = random.sample(['negative', 'positive'], 1)[0]
                prompt = demonstration + 'Review: ' + sample + '\n' + 'Question: Is the sentiment of the above review Positive or Negative?\n' + 'Answer: ' + random_label
                prompt_list.append(prompt)
        if format_index == 8:
            for sample in sampled_valid_samples:
                random_label = random.sample(['good', 'bad'], 1)[0]
                prompt = demonstration + 'My review for last night\'s ﬁlm: ' + sample + ' The critics agreed that this movie was ' + random_label
                prompt_list.append(prompt)
        return prompt_list

def gen_test_data_order(dataset_name, order_index = None):
    if dataset_name == 'ag_news':
        prompt_list = []
        dataset_test = load_dataset(dataset_name, split="test")[:2000]
        test_text = dataset_test['text']
        test_labels = dataset_test['label']
        test_sentences = test_text
        # generate demonstration examples
        examples = load_dataset(dataset_name, split="train")
        examples_text = examples['text']
        examples_labels = examples['label']
        examples_answers = ["world" if i == 0 else 'sports' if i==1 else 'business' if i == 2 else 'technology & science' for i in examples_labels]
        # examples_answers = ["world" if i == 0 else 'sports' if i==1 else 'business' if i == 2 else 'technology' for i in examples_labels]
        example_pairs = list(zip(examples_text, examples_answers))
        example_0_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 0]
        example_1_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 1]
        example_2_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 2]
        example_3_index = [i for i in range(len(examples_labels)) if examples_labels[i] == 3]
        rand_example_0_index = random.sample(example_0_index, 1)
        rand_example_1_index = random.sample(example_1_index, 1)
        rand_example_2_index = random.sample(example_2_index, 1)
        rand_example_3_index = random.sample(example_3_index, 1)
        rand_example_sample_index = [elem for pair in zip(rand_example_0_index, rand_example_1_index, rand_example_2_index, rand_example_3_index) for elem in pair]
        rand_example_sample_index_iterate = list(itertools.permutations(rand_example_sample_index))
        rand_example_sample_index_order = rand_example_sample_index_iterate[order_index]
        demonstration = ''
        for rand_index in rand_example_sample_index_order:
            example_i = 'Article: ' + example_pairs[rand_index][0] + '\n' + 'Answer: ' + example_pairs[rand_index][1] + '\n\n'
            demonstration += example_i
        for text in test_text:
            prompt = demonstration + 'Article: ' + text + '\n' + 'Answer:'
            prompt_list.append(prompt)
    return prompt_list, test_labels, demonstration, test_sentences, rand_example_sample_index_order

def gen_validate_data_order(dataset_name, demonstration=None):
    if dataset_name == 'ag_news':
        prompt_list = []
        dataset_validate = load_dataset(dataset_name, split="train")
        valid_text = dataset_validate['text']
        valid_labels = dataset_validate['label']
        valid_0_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 0]
        valid_1_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 1]
        valid_2_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 2]
        valid_3_index = [i for i in range(len(valid_labels)) if valid_labels[i] == 3]
        random_0_index = random.sample(valid_0_index, 20)
        random_1_index = random.sample(valid_1_index, 20)
        random_2_index = random.sample(valid_2_index, 20)
        random_3_index = random.sample(valid_3_index, 20)
        random_sample_index = sorted(random_0_index + random_1_index + random_2_index + random_3_index)
        sampled_valid_samples = [valid_text[i] for i in random_sample_index]
        for sample in sampled_valid_samples:
            random_label = random.sample(['world', 'sports', 'business', 'technology & science'],1)[0]
            # random_label = random.sample(['world', 'sports', 'business', 'technology'], 1)[0]
            prompt = demonstration + 'Article: ' + sample + '\n' + 'Answer: ' + random_label
            prompt_list.append(prompt)
        return prompt_list

# for Llama tokenizer, a token can repeat multiple times in the vocabulary
def find_possible_ids_for_labels(arg_str_list, tokenizer):
    # Initialize a dictionary to hold the IDs for each arg_str
    ids_dict = {arg_str[0]: [] for arg_str in arg_str_list}

    # Iterate over the range of IDs only once
    for id in range(32000):
        decoded = tokenizer.decode(id)

        # Check each arg_str for a match
        if decoded:
            for arg_str_tuple in arg_str_list:
                for arg_str in arg_str_tuple:
                    decoded = decoded.lower()
                    arg_str = arg_str.lower()
                    if len(arg_str) > 1:
                        if decoded in arg_str and arg_str[0] == decoded[0] and len(decoded)>1:
                            ids_dict[arg_str_tuple[0]].append(id)
                    else:
                        if decoded in arg_str and arg_str[0] == decoded[0]:
                            ids_dict[arg_str_tuple[0]].append(id)

    # Convert the dictionary to a list of lists for the IDs of each arg_str
    ids_list = list(ids_dict.values())
    max_len = max(len(sublist) for sublist in ids_list)
    padded_lst = [sublist + [sublist[0]] * (max_len - len(sublist)) for sublist in ids_list]
    return padded_lst

def ceil(number):
    return int(number + 1) if int(number) > 1 else 1

def sample_random_texts(texts, n_sample=5, seed=0, random_type="random_in_domain_words", cf_tokens=['N/A', 'null', '[MASK]'], length_ratio=1):
    """
    Construct content-free texts for estimating the model's prior.
    @params:
    texts: task corpus
    @Supporting content_free text types
    Assume the average length of the input texts is L.
    1. random_type="content_free_token": use L pre-defined content-free tokens to construct content_free texts
    2. random_type="random_english_words": use L random English words to construct content_free texts
    3. random_type="random_sentence": use random English sentences as content_free texts
    4. random_type="random_in_domain_words": use L random words sampled from the task corpus to construct content_free texts
    """
    np.random.seed(seed)
    is_sentence_pair = isinstance(texts[0], list)
    gen = DocumentGenerator()
    if not is_sentence_pair:
        all_words = []
        text_lengths = []
        for text in texts:
            words = text.lower().split()
            all_words = all_words + words
            text_lengths.append(len(words))
        ave_length = int(np.mean(text_lengths))
        random_texts = []
        if random_type == 'content_free_token':
            for cf_token in cf_tokens:
                random_texts.append(" ".join([cf_token]*ceil(length_ratio *ave_length))+" .")
        else:
            for i in range(n_sample):
                if random_type == 'random_english_words':
                    random_texts.append(" ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_length))) + " .")
                elif random_type == 'random_sentence':
                    random_texts.append(gen.sentence())
                else:
                    random_texts.append(" ".join(np.random.choice(all_words, size=ceil(length_ratio * ave_length)).tolist()) + " .")
    else:
        all_words_1 = []
        all_words_2 = []
        text_lengths_1 = []
        text_lengths_2 = []
        for text in texts:
            words_1 = text[0].lower().split()
            words_2 = text[1].lower().split()
            all_words_1 += words_1
            all_words_2 += words_2
            text_lengths_1.append(len(words_1))
            text_lengths_2.append(len(words_2))
        ave_l_1 = int(np.mean(text_lengths_1))
        ave_l_2 = int(np.mean(text_lengths_2))
        random_texts = []
        if random_type == 'content_free_token':
            for cf_token in cf_tokens:
                random_texts.append([" ".join([cf_token]*ceil(length_ratio * ave_l_1))+" .", " ".join([cf_token]*ceil(length_ratio * ave_l_2))+" ."])
        else:
            for i in range(n_sample):
                if random_type == 'random_english_words':
                    random_texts.append([" ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_l_1))) + " .", " ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_l_2))) + " ."])
                elif random_type == 'random_sentence':
                    random_texts.append([gen.sentence(), gen.sentence()])
                else:
                    random_texts.append([" ".join(np.random.choice(all_words_1, size=ceil(length_ratio * ave_l_1)).tolist()) + " .", " ".join(np.random.choice(all_words_2, size=ceil(length_ratio * ave_l_2)).tolist()) + " ."])
    return random_texts

def sample_random_texts(texts, n_sample=5, seed=0, random_type="random_in_domain_words", cf_tokens=['N/A', 'null', '[MASK]'], length_ratio=1):
    """
    Construct content-free texts for estimating the model's prior.
    @params:
    texts: task corpus
    @Supporting content_free text types
    Assume the average length of the input texts is L.
    1. random_type="content_free_token": use L pre-defined content-free tokens to construct content_free texts
    2. random_type="random_english_words": use L random English words to construct content_free texts
    3. random_type="random_sentence": use random English sentences as content_free texts
    4. random_type="random_in_domain_words": use L random words sampled from the task corpus to construct content_free texts
    """
    np.random.seed(seed)
    is_sentence_pair = isinstance(texts[0], list)
    gen = DocumentGenerator()
    if not is_sentence_pair:
        all_words = []
        text_lengths = []
        for text in texts:
            words = text.lower().split()
            all_words = all_words + words
            text_lengths.append(len(words))
        ave_length = int(np.mean(text_lengths))
        random_texts = []
        if random_type == 'content_free_token':
            for cf_token in cf_tokens:
                random_texts.append(" ".join([cf_token]*ceil(length_ratio *ave_length))+" .")
        else:
            for i in range(n_sample):
                if random_type == 'random_english_words':
                    random_texts.append(" ".join(np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_length))) + " .")
                elif random_type == 'random_sentence':
                    random_texts.append(gen.sentence())
                else:
                    random_texts.append(" ".join(np.random.choice(all_words, size=ceil(length_ratio * ave_length)).tolist()) + " .")
    else:
        all_words = [[] for _ in texts[0]]  # A list of lists to hold words for each sentence position
        text_lengths = [[] for _ in texts[0]]  # A list of lists to hold lengths for each sentence position

        # Splitting words and calculating text lengths for each position in the sentence lists
        for text in texts:
            for i, sentence in enumerate(text):
                words = sentence.lower().split()
                if i < len(all_words):
                    all_words[i] += words
                    text_lengths[i].append(len(words))

        ave_lengths = [int(np.mean(lengths)) for lengths in text_lengths]  # Average lengths for each sentence position
        random_texts = []

        for _ in range(n_sample):
            temp_text = []
            for i in range(len(texts[0])):  # Assuming all texts have the same length as the first one
                if random_type == 'content_free_token':
                    for cf_token in cf_tokens:
                        temp_text.append(" ".join([cf_token] * ceil(length_ratio * ave_lengths[i])) + " .")
                elif random_type == 'random_english_words':
                    temp_text.append(" ".join(
                        np.random.choice(sorted(english_words_set), size=ceil(length_ratio * ave_lengths[i]))) + " .")
                elif random_type == 'random_sentence':
                    temp_text.append(gen.sentence())
                else:  # random_in_domain_words
                    temp_text.append(" ".join(np.random.choice(all_words[i], size=ceil(length_ratio * ave_lengths[i])).tolist()) + " .")
            random_texts.append(temp_text)

    return random_texts

def write_json(outputfile, content):
    with open(outputfile, 'a') as f:
        f.write(json.dumps(content))
        f.write('\n')


# data, _, demonstration, test_sentences = prepare_dataset_test('wic')
# rand_text = sample_random_texts(texts=test_sentences, n_sample=20, seed=0)
# data_valid = prepare_dataset_validate('multi_nli', demonstration)
# prepare_dataset_validate('sst2')





