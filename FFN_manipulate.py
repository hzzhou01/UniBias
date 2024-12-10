import os
import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def biased_FFN_identify_and_eliminate(model, tokenizer, validate_data, ans_token_list, dataset_name):
    '''Identify and remove biased FFN neurons'''
    # get possible token ids for each lable
    gt_ans_ids_list = find_possible_ids_for_multi_str(ans_token_list, tokenizer)
    ans_ids_list = gt_ans_ids_list
    # get the logit of label tokens for each value vector
    logits, logits_sum = find_value_logits(model, torch.tensor(ans_ids_list))
    grid_biased_neuron_list_org = []
    for th in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for bias_th in np.arange(0.1, 0.51, 0.1):
            biased_neuron_dict = find_biased_FFN_neuron_candidates(logits, logits_sum, th=th, bias_th=bias_th) # FFN neurons that are biased towards certain labels when projected to the vocabulary space
            grid_biased_neuron_list_org.append(biased_neuron_dict)
    grid_biased_neuron_list = remove_identical_sublist(grid_biased_neuron_list_org)
    grid_search_biased_FFNs_list_org = []
    for biased_neuron_dict in grid_biased_neuron_list:
        grid_search_biased_FFNs_list_i = find_biased_FFN_neurons(model, tokenizer, biased_neuron_dict, validate_data, ans_token_list, dataset_name)
        grid_search_biased_FFNs_list_org += grid_search_biased_FFNs_list_i
    grid_search_biased_FFNs_list = remove_identical_sublist(grid_search_biased_FFNs_list_org)
    if {} not in grid_search_biased_FFNs_list:
        grid_search_biased_FFNs_list.append({})
    org_logit_bias, org_label_logit = logit_bias_estimate(model, tokenizer, {}, validate_data, gt_ans_ids_list, ans_token_list, dataset_name, 0)
    org_label_logit_sum = sum(org_label_logit)
    grid_search_logit_bias, grid_search_label_logit, biased_FFNs, debias_alpha_value, filtered_grid_search_FFNs_list = grid_search_biased_FFNs(model, tokenizer, grid_search_biased_FFNs_list, validate_data, ans_token_list, dataset_name, org_label_logit_sum)

    min_logit_bias_index = grid_search_logit_bias.index(min(grid_search_logit_bias))
    min_bias_label_logit = grid_search_label_logit[min_logit_bias_index]
    min_bias = grid_search_logit_bias[min_logit_bias_index]
    new_label_logit_sum = max(sum(min_bias_label_logit), org_label_logit_sum)
    # if bias remains large, try alpha values other than 0
    if min_bias > 0.1:
        topk_num = min(5, len(grid_search_logit_bias))
        topk_index = torch.topk(torch.tensor(grid_search_logit_bias), largest=False, k=topk_num)[1]
        topk_biased_FFNs_list = [filtered_grid_search_FFNs_list[i] for i in topk_index if filtered_grid_search_FFNs_list[i] !={}]
        grid_search_logit_bias_2, grid_search_logit_2, biased_FFNs_2, debias_alpha_value_2, _ = grid_search_biased_FFNs(model,
                                                                                                           tokenizer,
                                                                                                           topk_biased_FFNs_list,
                                                                                                           validate_data,
                                                                                                           ans_token_list,
                                                                                                           dataset_name,
                                                                                                           new_label_logit_sum,
                                                                                                           debias_alpha_list=[-0.5, -1, -2, -4])
        if biased_FFNs_2:
            min_logit_bias_index_2 = grid_search_logit_bias_2.index(min(grid_search_logit_bias_2))
            min_bias_label_logit_2 = grid_search_logit_2[min_logit_bias_index_2]
            min_bias_2 = grid_search_logit_bias_2[min_logit_bias_index_2]
            if min_bias_2 < min_bias:
                biased_FFNs = biased_FFNs_2
                min_bias_label_logit = min_bias_label_logit_2
                debias_alpha_value = debias_alpha_value_2
    hooks = set_value_activations(model, biased_FFNs, coef_value=debias_alpha_value)
    return biased_FFNs, min_bias_label_logit, debias_alpha_value

def find_value_logits(model, ans_ids_list):
    '''get the logits of label tokens for each value vector'''
    logits = []
    logits_sum = []
    for i in tqdm(range(model.config.num_hidden_layers)):

        layer_logits = model.lm_head(model.model.norm(model.model.layers[i].mlp.down_proj.weight.T))
        layer_logits = F.softmax(layer_logits, dim=-1)
        ans_token_logits = layer_logits[:, ans_ids_list]
        ans_token_logits_max, _ = ans_token_logits.max(dim=2)
        ans_token_logits_sum = torch.sum(ans_token_logits_max, dim=-1)
        logits.append(ans_token_logits_max.detach().cpu())
        logits_sum.append(ans_token_logits_sum.detach().cpu())
    return logits, logits_sum

def find_biased_FFN_neuron_candidates(logits, logits_sum, th = 0.1, bias_th = 0.1 ):
    '''Find FFN value vectors that are biased towards certain labels when projected to the vocabulary space (bias criterion)'''
    biased_neuron_dict = {}
    for layer_index in range(len(logits)):
        layer_logits = logits[layer_index]
        layer_logits_sum = logits_sum[layer_index]
        biased_indexes = torch.where(layer_logits_sum > th)[0]
        if len(biased_indexes) > 0:
            for i in biased_indexes:
                biased_logit_candidate = layer_logits[i].to(dtype=torch.float32)
                logit_bias = logit_bias_measure(biased_logit_candidate)
                if logit_bias > bias_th:
                    if layer_index in biased_neuron_dict.keys():
                        biased_neuron_dict[layer_index].append(i.item())
                    else:
                        biased_neuron_dict[layer_index] = [i.item()]
    return biased_neuron_dict

def find_biased_FFN_neurons(model, tokenizer, biased_neuron_dict, validate_data, ans_token_list, dataset_name):
    '''Find FFN value vectors by relatedness criterion and low-variance criterion'''
    all_biased_coefficients = []
    for layer_i in biased_neuron_dict.keys():
        for indices in biased_neuron_dict[layer_i]:
            all_biased_coefficients.append([])
    for index, gt_text in enumerate(validate_data):
        gts = tokenizer(gt_text, return_tensors="pt").to(model.device)
        gt_ids = gts["input_ids"]
        gt_tokens = [tokenizer.decode(gt_ids[0][i], skip_special_tokens=True) for i in range(gt_ids.shape[-1])]
        gt_ans_index = find_answer_location(gt_tokens, task = dataset_name)
        ans_token = gt_tokens[gt_ans_index]
        if is_ans_in_anslist(ans_token, ans_token_list) is False:
            print('error in finding answer')

        output_coefficients = []
        def capture_coefficients_hook(module, input, output):
            # This will store the coefficients in coefficient
            output_coefficients.append(input[0].detach())

        coefficient_hooks = []
        for layer_index in range(model.config.num_hidden_layers):
            coefficient_hook = model.model.layers[layer_index].mlp.down_proj.register_forward_hook(
                capture_coefficients_hook)
            coefficient_hooks.append(coefficient_hook)

        with torch.no_grad():
            outputs = model(gt_ids, output_hidden_states=False, output_attentions=False)

        for coefficient_hook in coefficient_hooks:
            coefficient_hook.remove()

        sample_biased_coefficients = []
        for layer_i in biased_neuron_dict.keys():
            for indices in biased_neuron_dict[layer_i]:
                sample_biased_coefficients.append(output_coefficients[layer_i][0][:,indices][gt_ans_index-1].detach().cpu())
        for sub_list, element in zip(all_biased_coefficients, sample_biased_coefficients):
            sub_list.append(element)
    all_biased_coefficients_ave = [torch.mean(torch.stack(sub_list, dim=0)) for sub_list in all_biased_coefficients]
    all_biased_coefficients_cv = [torch.std(torch.stack(sub_list, dim=0)) / torch.mean(torch.stack(sub_list, dim=0)) for sub_list in all_biased_coefficients]
    grid_search_biased_FFNs_list = []
    for th_cv in np.arange(0.05, 0.21, 0.05): # low-variance criterion
        for th_ave in [0.1, 0.2, 0.3]: # relatedness croterion
            i = 0
            filtered_biased_FFN_neurons = {}
            for layer_i in biased_neuron_dict.keys():
                for indices in biased_neuron_dict[layer_i]:
                    if abs(all_biased_coefficients_cv[i]) < th_cv and abs(all_biased_coefficients_ave[i]) > th_ave:
                        if layer_i in filtered_biased_FFN_neurons.keys():
                            filtered_biased_FFN_neurons[layer_i].append(indices)
                        else:
                            filtered_biased_FFN_neurons[layer_i] = [indices]
                    i += 1
            grid_search_biased_FFNs_list.append(filtered_biased_FFN_neurons)
    return grid_search_biased_FFNs_list

def grid_search_biased_FFNs(model, tokenizer, grid_search_FFNs_list, validate_data, ans_token_list, dataset_name, org_label_logit_sum = 0, debias_alpha_list = [0.]):
    '''grid search thresholds to select the biased FFN vectors'''
    gt_ans_ids_list = find_possible_ids_for_multi_str(ans_token_list, tokenizer)
    grid_search_logit_bias = []
    grid_search_logit = []
    filtered_grid_search_FFNs_list = []
    min_ave_logit_bias = None
    biased_FFN_neurons_record, debias_alpha_record = None, None
    for debias_alpha in debias_alpha_list:
        for biased_FFN_neurons in grid_search_FFNs_list:
            ave_logit_bias, ave_logit = logit_bias_estimate(model, tokenizer, biased_FFN_neurons, validate_data, gt_ans_ids_list, ans_token_list, dataset_name, debias_alpha)
            if sum(ave_logit) > org_label_logit_sum * 0.9:
                grid_search_logit_bias.append(ave_logit_bias)
                grid_search_logit.append(ave_logit)
                filtered_grid_search_FFNs_list.append(biased_FFN_neurons)
                if min_ave_logit_bias:
                    if ave_logit_bias < min_ave_logit_bias:
                        min_ave_logit_bias = ave_logit_bias
                        biased_FFN_neurons_record = biased_FFN_neurons
                        debias_alpha_record = debias_alpha
                else:
                    min_ave_logit_bias = ave_logit_bias
                    biased_FFN_neurons_record = biased_FFN_neurons
                    debias_alpha_record = debias_alpha
    return grid_search_logit_bias, grid_search_logit, biased_FFN_neurons_record, debias_alpha_record, filtered_grid_search_FFNs_list

def find_possible_ids_for_multi_str(arg_str_list, tokenizer):
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

def logit_bias_measure(label_logit_list):
    total_sum = sum(label_logit_list)
    ave = total_sum/len(label_logit_list)
    return sum([abs((ave-i)/ave) for i in label_logit_list])/len(label_logit_list)

def set_value_activations(model, values_per_layer, coef_value=0):
    """
    Uses PyTorch hooks to set the activations of each value in values_per_layer to coef_value
    The modeling_llama.py in transformers need to be changed to allow masking coefficiets:
    Replacing
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    With:
    coeeficients = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    down_proj = self.down_proj(coeeficients)
    """

    def value_activation_replacement_hook(values, coef_val):
        def hook(module, input, output):
            output[:, :, values] = coef_val

        return hook

    hooks = []
    NB_LAYERS = len(model.model.layers)
    for layer in range(NB_LAYERS):
        if layer in values_per_layer:
            values = values_per_layer[layer]
        else:
            values = []

        hook = model.model.layers[layer].mlp.up_proj.register_forward_hook(
            value_activation_replacement_hook(values, coef_value)
        )

        hooks.append(hook)

    hooks = hooks

    return hooks

def remove_all_hooks(hooks):
    if hooks is not None:
        for hook in hooks:
            hook.remove()

        hooks = None
    else:
        print("No hooks to remove")

def find_answer_location(full_tokens, task = 'sst2'):
    for i in range(5,len(full_tokens)):
        if task in ('sst2', 'cr', 'mr', 'sst5'):
            if full_tokens[i-3] == 'S' and full_tokens[i-2] == 'ent' and full_tokens[i-1] == 'iment' and full_tokens[i] == ':':
                index = i+1
        elif task == 'copa':
            if full_tokens[i - 1] == 'Answer' and full_tokens[i] == ':':
                index = i+2 # llama output '' before numbers (1,2)
        elif task == 'trec':
            if full_tokens[i - 2] == 'Answer' and full_tokens[i - 1] == 'Type' and full_tokens[i] == ':':
                index = i+1
        else:
            if full_tokens[i - 1] == 'Answer' and full_tokens[i] == ':':
                index = i+1
    return index#, answer_token

def remove_identical_sublist(biased_list):
    # remove identical sublists, which are dicts
    unique_json = set(json.dumps(dict_item, sort_keys=True) for dict_item in biased_list)
    unique_biased_list = [{int(key): value for key, value in json.loads(item).items()} for item in unique_json]
    return unique_biased_list

def is_ans_in_anslist(ans, ans_list):
    if ans:
        for ans_sublist in ans_list:
            for ans_i in ans_sublist:
                if ans in ans_i:
                    return True
    return False

def logit_bias_estimation(label_logit_list):
    total_sum = sum(label_logit_list)
    ave = total_sum/len(label_logit_list)
    # return sum([abs((ave - i) / ave) for i in label_logit_list]) / len(label_logit_list)
    return sum([abs(np.log(ave)-np.log(i)) for i in label_logit_list])/len(label_logit_list)

def logit_bias_estimate(model, tokenizer, filtered_biased_FFN_neurons, prompt_list, gt_ans_ids_list, ans_token_list, dataset_name, debias_alpha, prob_bool=True):
    hooks = set_value_activations(model, filtered_biased_FFN_neurons, coef_value=debias_alpha)
    sample_logit_bias_list = []
    all_valid_probs = [[] for i in range(len(gt_ans_ids_list))]
    for index, gt_text in enumerate(prompt_list):
        gts = tokenizer(gt_text, return_tensors="pt").to(model.lm_head.weight.device)
        gt_ids = gts["input_ids"]
        gt_tokens = [tokenizer.decode(gt_ids[0][i], skip_special_tokens=True) for i in range(gt_ids.shape[-1])]
        gt_ans_index = find_answer_location(gt_tokens, task=dataset_name)
        ans_token = gt_tokens[gt_ans_index]
        if is_ans_in_anslist(ans_token, ans_token_list) is False:
            print('error in finding answer')

        with torch.no_grad():
            outputs = model(gt_ids, output_hidden_states=False, output_attentions=False)
            output_logit = outputs.logits.detach().cpu()[0]
            gt_all_probs=[]
            for gt_ans_ids in gt_ans_ids_list:
                if gt_ans_ids:
                    gt_i_probs = []
                    for id_i in gt_ans_ids:
                        aux_index = output_logit[gt_ans_index - 1]  # -1 because the previous token is making prediction of arguments
                        if prob_bool:
                            aux_index = F.softmax(aux_index, dim=-1)
                        prob_gt_i = aux_index[id_i].to(dtype=torch.float32).cpu().numpy()
                        gt_i_probs.append(prob_gt_i)
                    max_index = max(enumerate(gt_i_probs), key=lambda x: abs(x[1]))[0]
                    gt_all_probs.append(gt_i_probs[max_index])
            for sublist, element in zip(all_valid_probs, gt_all_probs):
                sublist.append(element)
    ave_sample_logit_bias = logit_bias_estimation([sum(sublist) / len(sublist) for sublist in all_valid_probs])
    remove_all_hooks(hooks)
    return ave_sample_logit_bias, [sum(sublist) / len(sublist) for sublist in all_valid_probs]
