
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import math
import random
import sklearn
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup


MAX_LEN = 64
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir):

    with open(article_dir) as json_file:
        article_json = json.load(json_file)

    information_bias_label = []

    for i in range(len(article_json['body'])):
        sentence_text = article_json['body'][i]['sentence']
        sentence_encoding = tokenizer.encode_plus(sentence_text, return_tensors='pt', add_special_tokens=True,
                                                  max_length=MAX_LEN, padding='max_length', truncation=True,
                                                  return_token_type_ids=True, return_attention_mask=True)

        sentence_input_ids = sentence_encoding['input_ids']
        sentence_token_type_ids = sentence_encoding['token_type_ids']
        sentence_attention_mask = sentence_encoding['attention_mask']

        if i == 0:
            all_sentence_input_ids = sentence_input_ids
            all_sentence_token_type_ids = sentence_token_type_ids
            all_sentence_attention_mask = sentence_attention_mask
        else:
            all_sentence_input_ids = torch.cat((all_sentence_input_ids, sentence_input_ids), dim=0)
            all_sentence_token_type_ids = torch.cat((all_sentence_token_type_ids, sentence_token_type_ids), dim=0)
            all_sentence_attention_mask = torch.cat((all_sentence_attention_mask, sentence_attention_mask), dim=0)

        sentence_info_bias_label = 0
        if len(article_json['body'][i]['annotations']) != 0:
            for j in range(len(article_json['body'][i]['annotations'])):
                if article_json['body'][i]['annotations'][j]['bias'] == 'Informational':
                    sentence_info_bias_label = 1

        information_bias_label.append(sentence_info_bias_label)

    information_bias_label = torch.tensor(information_bias_label, dtype=torch.long)

    article_discourse = pd.read_csv(article_discourse_dir, sep='\t', index_col=0)
    discourse_prob_label = article_discourse.to_numpy()
    discourse_prob_label = torch.tensor(discourse_prob_label, dtype=torch.float) # number of sentences * 9

    article_comp = pd.read_csv(article_comp_dir, sep = '\t', index_col=0)
    comp_prob_label = article_comp.to_numpy()
    comp_prob_label = torch.tensor(comp_prob_label, dtype=torch.float)  # number of sentences * 2

    article_cont = pd.read_csv(article_cont_dir, sep='\t', index_col=0)
    cont_prob_label = article_cont.to_numpy()
    cont_prob_label = torch.tensor(cont_prob_label, dtype=torch.float)  # number of sentences * 2


    return all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label









# stop here
