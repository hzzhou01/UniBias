import json
import torch
import torch.nn.functional as F
import random
import statistics
from tqdm import tqdm
import numpy as np

def attention_manipulate(model, tokenizer, validate_data, ans_token_list, dataset_name):
    '''Identify and eliminate biased attention heads '''
    # get possible token ids for each lable
    gt_ans_ids_list = find_possible_ids_for_multi_str(ans_token_list, tokenizer)
    grid_search_biased_AHs_list_org, grid_search_threshold_list = biased_attention_head_identification(model, tokenizer, validate_data, ans_token_list, dataset_name)
    org_logit_bias, org_label_logit = logit_bias_estimate(model, tokenizer, {}, validate_data, gt_ans_ids_list, ans_token_list, dataset_name, debias_alpha=0)
    org_label_logit_sum = sum(org_label_logit)
    grid_search_biased_AHs_list = remove_identical_sublist(grid_search_biased_AHs_list_org)
    grid_search_logit_bias, grid_search_logit, biased_AHs, debias_alpha_value, filtered_grid_search_AHs_list = grid_search_biased_AHs(model, tokenizer, grid_search_biased_AHs_list, validate_data, ans_token_list, dataset_name, org_label_logit_sum, debias_alpha_list=[0.])

    min_logit_bias_index = grid_search_logit_bias.index(min(grid_search_logit_bias))
    min_bias_label_logit = grid_search_logit[min_logit_bias_index]
    min_bias = grid_search_logit_bias[min_logit_bias_index]
    new_label_logit_sum = max(sum(min_bias_label_logit), org_label_logit_sum)
    debias_alpha_value = float(debias_alpha_value)
    # if bias remains large, try alpha values other than 0
    if min_bias > 0.1:
        topk_num = min(5, len(grid_search_logit_bias))
        topk_index = torch.topk(torch.tensor(grid_search_logit_bias), largest=False, k=topk_num)[1]
        topk_biased_AHs_list = [filtered_grid_search_AHs_list[i] for i in topk_index if filtered_grid_search_AHs_list[i] !={}]
        grid_search_logit_bias_2, grid_search_logit_2, biased_AHs_2, debias_alpha_value_2, _ = grid_search_biased_AHs(model,
                                                                                                           tokenizer,
                                                                                                           topk_biased_AHs_list,
                                                                                                           validate_data,
                                                                                                           ans_token_list,
                                                                                                           dataset_name,
                                                                                                           new_label_logit_sum,
                                                                                                           debias_alpha_list=[-0.1, -0.25, -0.5, -1])
        if biased_AHs_2:
            min_logit_bias_index_2 = grid_search_logit_bias_2.index(min(grid_search_logit_bias_2))
            min_bias_label_logit_2 = grid_search_logit_2[min_logit_bias_index_2]
            min_bias_2 = grid_search_logit_bias_2[min_logit_bias_index_2]
            if min_bias_2 < min_bias:
                biased_AHs = biased_AHs_2
                min_bias_label_logit = min_bias_label_logit_2
                debias_alpha_value = debias_alpha_value_2
    # mask biased attention heads
    set_attention_masks(model, biased_AHs, debias_alpha_value)
    return biased_AHs, min_bias_label_logit, debias_alpha_value

def is_ans_in_anslist(ans, ans_list):
    if ans:
        for ans_sublist in ans_list:
            for ans_i in ans_sublist:
                if ans in ans_i:
                    return True
    return False

def biased_attention_head_identification(model, tokenizer, validate_data, ans_token_list, dataset_name):
    """Identify biased attention heads candidates based on three criterions"""
    NB_LAYERS = len(model.model.layers)
    NB_HEADS = model.model.layers[0].self_attn.num_heads
    _cumulate_all_attention_weights = [[[] for _ in range(NB_HEADS)] for _ in range(NB_LAYERS)]
    _cumulate_all_head_logit_record = [[[] for _ in range(NB_HEADS)] for _ in range(NB_LAYERS - 1)]
    _cumulate_all_hidden_logit_record = [[[] for _ in range(NB_HEADS)] for _ in range(NB_LAYERS - 1)]

    gt_ans_ids_list = find_possible_ids_for_multi_str(ans_token_list, tokenizer)
    for index in tqdm(range(len(validate_data))):
        gt_text = validate_data[index]
        gts = tokenizer(gt_text, return_tensors="pt").to(model.lm_head.weight.device)
        gt_ids = gts["input_ids"]
        gt_tokens = [tokenizer.decode(gt_ids[0][i], skip_special_tokens=True) for i in range(gt_ids.shape[-1])]
        gt_ans_index = find_answer_location(gt_tokens, task=dataset_name)
        gt_ans_index_reverse = gt_ans_index - len(gt_tokens)
        ans_token = gt_tokens[gt_ans_index]
        ans_token_reverse = gt_tokens[gt_ans_index_reverse]
        if is_ans_in_anslist(ans_token, ans_token_list) is False:
            print('error in finding answer')

        with torch.no_grad():
            torch.cuda.empty_cache()

            hidden_states_attention = [] # the output hidden stae=tes of each attention head
            def capture_head_output_hook(module, input, output):
                hidden_states_attention.append(output.detach().cpu()[:,:,-10:,])

            head_output_hooks = []
            for layer_index in range(model.config.num_hidden_layers):
                hook = model.model.layers[layer_index].self_attn.custom_head_output.register_forward_hook(capture_head_output_hook)
                head_output_hooks.append(hook)

            outputs = model(gt_ids, output_hidden_states=True, output_attentions=False)
            hidden_states = outputs.hidden_states[1:]

            for head_output_hook in head_output_hooks:
                head_output_hook.remove()

        for layer_index in range(1, len(hidden_states_attention)):
            attentions_layer_i = hidden_states_attention[layer_index][0]
            hidden_states_layer_i_1 = hidden_states[layer_index-1][0]

            layer_i_label_logits_record, layer_i_label_hidden_logits_record = calculate_bias_logits_by_attention_heads(
                model, attentions_layer_i, hidden_states_layer_i_1,
                # gt_ans_index,
                gt_ans_index_reverse, #only outputs final 10 attention head logits, so use reverse
                gt_ans_ids_list)
            for head_i_index, head_i_logit in enumerate(layer_i_label_logits_record):
                _cumulate_all_head_logit_record[layer_index - 1][head_i_index].append(head_i_logit)
            for head_i_index, hidden_logit in enumerate(layer_i_label_hidden_logits_record):
                _cumulate_all_hidden_logit_record[layer_index - 1][head_i_index].append(hidden_logit)
        del hidden_states, hidden_states_attention, attentions_layer_i, hidden_states_layer_i_1

    _cumulate_all_head_logit_record_transpose = np.array(_cumulate_all_head_logit_record).transpose(0, 1, 3, 2).tolist()
    _cumulate_all_hidden_logit_record_transpose = np.array(_cumulate_all_hidden_logit_record).transpose(0, 1, 3, 2).tolist()
    _cumulate_all_hidden_logit_ave = [
        [
            [
                statistics.mean(label_i_hidden_logit_list) if label_i_hidden_logit_list else None
                for label_i_hidden_logit_list in head_hidden_logit_list
            ]
            for head_hidden_logit_list in layer_hidden_logit_list
        ]
        for layer_hidden_logit_list in _cumulate_all_hidden_logit_record_transpose
    ]

    _cumulate_all_head_logit_ave = [
        [
            [
                (statistics.mean(label_i_logit_list)) if label_i_logit_list else None
                for label_i_logit_list in head_i_logit_list
            ]
            for head_i_logit_list in layer_head_logit_list
        ]
        for layer_head_logit_list in _cumulate_all_head_logit_record_transpose
    ]

    _cumulate_all_head_logit_cv = [
        [
            [
                (statistics.stdev(label_i_logit_list) + 1e-10) / (
                            (statistics.mean(label_i_logit_list)) + 1e-10) if len(label_i_logit_list) > 1 else 0
                for label_i_logit_list in head_i_logit_list
            ]
            for head_i_logit_list in layer_head_logit_list
        ]
        for layer_head_logit_list in _cumulate_all_head_logit_record_transpose
    ]

    _cumulate_all_head_logit_overall_cv = [
        [

            sum((abs(label_i_ave) / (sum(abs(ave) for ave in head_ave_list) + 1e-10)) * abs(label_i_cv) for
                label_i_cv, label_i_ave in zip(head_cv_list, head_ave_list))
            for head_cv_list, head_ave_list in zip(layer_cv_list, layer_ave_list)
        ]
        for layer_cv_list, layer_ave_list in zip(_cumulate_all_head_logit_cv, _cumulate_all_head_logit_ave)
    ]
    grid_search_biased_AHs_list = []
    grid_search_threshold_list = []
    for th_bias in np.arange(0.1, 0.51, 0.1): # bias criterion
        for th_sum in [0.02, 0.05, 0.1, 0.15, 0.2]: # relatedness criterion
            for th_cv in np.arange(0.1, 0.251, 0.05): # low-variance criterion
                bias_head_dict_i = filter_biased_attention_heads(_cumulate_all_head_logit_ave,
                                                                   _cumulate_all_hidden_logit_ave,
                                                                   _cumulate_all_head_logit_overall_cv,
                                                                   th_bias=th_bias,
                                                                   th_sum=th_sum,
                                                                   th_cv=th_cv)
                grid_search_biased_AHs_list.append(bias_head_dict_i)
                grid_search_threshold_list.append((th_bias, th_sum, th_cv))
    if {} not in grid_search_biased_AHs_list:
        grid_search_biased_AHs_list.append({})
        grid_search_threshold_list.append([])
    return grid_search_biased_AHs_list, grid_search_threshold_list

def remove_identical_sublist(biased_list):
    # remove identical sublists, which are dicts
    unique_json = set(json.dumps(dict_item, sort_keys=True) for dict_item in biased_list)
    unique_biased_list = [{int(key): value for key, value in json.loads(item).items()} for item in unique_json]
    return unique_biased_list

def find_all_indices(list1, list2):
    indices = []
    for item in list1:
        item_indices = [index for index, value in enumerate(list2) if value == item]
        indices.append(item_indices)
    return indices

def grid_search_biased_AHs(model, tokenizer, grid_search_AHs_list, validate_data, ans_token_list, dataset_name, org_label_logit_sum = 0, debias_alpha_list = [0.]):
    '''grid search thresholds to select the biased attention heads'''
    gt_ans_ids_list = find_possible_ids_for_multi_str(ans_token_list, tokenizer)
    grid_search_logit_bias = []
    grid_search_logit = []
    filtered_grid_search_AHs_list = []
    min_ave_logit_bias = None
    biased_AHs_dict_record = None
    debias_alpha_record = None
    for debias_alpha in debias_alpha_list:
        for biased_AHs_dict in grid_search_AHs_list:
            ave_logit_bias, ave_logit = logit_bias_estimate(model, tokenizer, biased_AHs_dict, validate_data, gt_ans_ids_list, ans_token_list, dataset_name, debias_alpha)
            if sum(ave_logit) > org_label_logit_sum * 0.95:
                grid_search_logit_bias.append(ave_logit_bias)
                grid_search_logit.append(ave_logit)
                filtered_grid_search_AHs_list.append(biased_AHs_dict)
                if min_ave_logit_bias:
                    if ave_logit_bias < min_ave_logit_bias:
                        min_ave_logit_bias = ave_logit_bias
                        biased_AHs_dict_record = biased_AHs_dict
                        debias_alpha_record = debias_alpha
                else:
                    min_ave_logit_bias = ave_logit_bias
                    biased_AHs_dict_record = biased_AHs_dict
                    debias_alpha_record = debias_alpha
    return grid_search_logit_bias, grid_search_logit, biased_AHs_dict_record, debias_alpha_record, filtered_grid_search_AHs_list

def logit_bias_estimation(label_logit_list):
    total_sum = sum(label_logit_list)
    ave = total_sum/len(label_logit_list)
    return sum([abs(np.log(ave)-np.log(i)) for i in label_logit_list]) / len(label_logit_list)

def logit_bias_estimate(model, tokenizer, biased_AHs_dict, prompt_list, gt_ans_ids_list, ans_token_list, dataset_name, debias_alpha, prob_bool=True):
    set_attention_masks(model, biased_AHs_dict, debias_alpha)
    sample_logit_bias_list = []
    all_valid_logits = [[] for i in range(len(gt_ans_ids_list))]
    for index, gt_text in enumerate(prompt_list):
        # gt_text = prompt_list[0]
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
            gt_all_logits=[]
            for gt_ans_ids in gt_ans_ids_list:
                if gt_ans_ids:
                    gt_i_logits = []
                    for id_i in gt_ans_ids:
                        aux_index = output_logit[gt_ans_index - 1]  # -1 because the previous token is making prediction of arguments
                        if prob_bool:
                            aux_index = F.softmax(aux_index, dim=-1)
                        prob_gt_i = aux_index[id_i].to(dtype=torch.float32).cpu().numpy()
                        gt_i_logits.append(prob_gt_i)
                    gt_all_logits.append(float(max(gt_i_logits)))
            for sublist, element in zip(all_valid_logits, gt_all_logits):
                sublist.append(element)
    ave_sample_logit_bias = logit_bias_estimation([sum(sublist) / len(sublist) for sublist in all_valid_logits])
    remove_attention_masks(model, biased_AHs_dict)
    return ave_sample_logit_bias, [sum(sublist) / len(sublist) for sublist in all_valid_logits]


def set_attention_masks(model, AHs_dict, debias_alpha):
    for layer in AHs_dict.keys():
        head_indexes = AHs_dict[layer]
        # head_weight = AHs_dict[layer][1]
        model.model.layers[int(layer)].self_attn.mask[0, head_indexes, 0, 0] = debias_alpha

def remove_attention_masks(model, AHs_dict):
    for layer in AHs_dict.keys():
        head_indexes = AHs_dict[layer]
        # head_weight = AHs_dict[layer][1]
        model.model.layers[int(layer)].self_attn.mask[0, head_indexes, 0, 0] = 1

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
    return index


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

def decode_gt_prob(model, output, gt_ans_ids_list, norm_bool, softmax_bool = True):
  """This function decodes the tokens and projets them into the vocabulary space"""
  with torch.no_grad():
      if norm_bool:
          aux_index = model.model.norm(output)
      else:
          aux_index = output
      aux_index = model.lm_head(aux_index.to(model.lm_head.weight.device)) #[0] # 32000
      if softmax_bool:
        aux_index = F.softmax(aux_index,dim = -1)
  gt_all_probs = []
  for gt_ans_ids in gt_ans_ids_list:
      if gt_ans_ids:
          gt_i_probs = []
          for id_i in gt_ans_ids:
            prob_gt_i = aux_index[id_i].to(dtype=torch.float32).cpu().numpy()
            gt_i_probs.append(prob_gt_i)
          gt_all_probs.append(float(max(gt_i_probs)))
  return gt_all_probs

def logit_bias_measure(label_logit_list):
    total_sum = sum(label_logit_list)
    ave = total_sum/len(label_logit_list)
    return sum([abs((ave-i)/ave) for i in label_logit_list])/len(label_logit_list)

def filter_biased_attention_heads(_cumulate_all_prob_change_ave, _cumulate_all_hidden_prob_change_ave, _cumulate_all_prob_change_overall_cv, th_bias, th_sum, th_cv):
    bias_head_dict = {}
    for layer_index in range(15, len(_cumulate_all_prob_change_ave)):
        layer_i_prob_change = torch.sum(torch.abs(torch.tensor(_cumulate_all_prob_change_ave)), dim=-1)[layer_index]
        layer_i_hidden_logit = torch.sum(torch.abs(torch.tensor(_cumulate_all_hidden_prob_change_ave)), dim=-1)[layer_index]
        for head_index in range(layer_i_prob_change.shape[0]):
            if layer_i_prob_change[head_index] > th_sum * layer_i_hidden_logit[head_index] and layer_i_prob_change[head_index] > 0.2:
                head_i_bias_estimation = logit_bias_measure(_cumulate_all_prob_change_ave[layer_index][head_index])
                if head_i_bias_estimation > th_bias:
                    if _cumulate_all_prob_change_overall_cv[layer_index][head_index] < th_cv and _cumulate_all_prob_change_overall_cv[layer_index][head_index] > 0:
                        if (layer_index +1) in bias_head_dict.keys():
                            bias_head_dict[layer_index + 1].append(head_index)
                        else:
                            bias_head_dict[layer_index + 1]=[head_index]
    return bias_head_dict

def calculate_bias_logits_by_attention_heads(model, attention_matrices, hidden_states_layer_i_1, ans_index, gt_ans_ids_list):
    num_heads = attention_matrices.shape[0]
    heads_label_logits_record = []
    layer_i_label_hidden_logits_record=[]

    head_i_label_hidden_logits = decode_gt_prob(model, hidden_states_layer_i_1[ans_index - 1], gt_ans_ids_list, norm_bool=False, softmax_bool=False)
    for i in range(num_heads):
        head_i_label_logits = decode_gt_prob(model, attention_matrices[i][ans_index-1], gt_ans_ids_list, norm_bool=False, softmax_bool=False)
        heads_label_logits_record.append(head_i_label_logits)
        layer_i_label_hidden_logits_record.append(head_i_label_hidden_logits)
    return heads_label_logits_record, layer_i_label_hidden_logits_record