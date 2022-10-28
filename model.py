
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



class Sentence_Words_Embeddings(nn.Module):
    # input: input_ids, token_type_ids, attention_mask
    # output: batch_size * number of tokens * 768
    def __init__(self):
        super(Sentence_Words_Embeddings, self).__init__()

        self.robertamodel = RobertaModel.from_pretrained("roberta-base", output_hidden_states = True, )

    def forward(self, input_ids, token_type_ids, attention_mask):

        outputs = self.robertamodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings_batch = torch.stack(hidden_states, dim = 0)

        feature_matrix_batch = list()

        for i in range(input_ids.shape[0]):
            token_embeddings = token_embeddings_batch[:,i,:,:]
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)

            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)
            token_vecs_sum = torch.stack(token_vecs_sum, dim=0) # number of tokens * 768, token embeddings within a sentence

            feature_matrix_batch.append(token_vecs_sum)

        feature_matrix_batch = torch.stack(feature_matrix_batch, dim = 0) # batch_size * number of tokens * 768

        return feature_matrix_batch




class Joint_Classifier(nn.Module):
    def __init__(self):
        super(Joint_Classifier, self).__init__()

        self.sentence_words_embeddings = Sentence_Words_Embeddings()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)

        self.info_bias_classifier_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.info_bias_classifier_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.info_bias_classifier_1.bias)

        self.info_bias_classifier_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.info_bias_classifier_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.info_bias_classifier_2.bias)

        self.discourse_classifier_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.discourse_classifier_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.discourse_classifier_1.bias)

        self.discourse_classifier_2 = nn.Linear(768, 9, bias=True)
        nn.init.xavier_uniform_(self.discourse_classifier_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.discourse_classifier_2.bias)

        self.crossentropyloss = nn.CrossEntropyLoss(reduction='sum')
        self.mseloss = nn.MSELoss(reduction='sum')

        self.pdtb_comparison_classifier_1 = nn.Linear(768 * 2, 768 * 2, bias=True)
        nn.init.xavier_uniform_(self.pdtb_comparison_classifier_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.pdtb_comparison_classifier_1.bias)

        self.pdtb_comparison_classifier_2 = nn.Linear(768 * 2, 2, bias=True)
        nn.init.xavier_uniform_(self.pdtb_comparison_classifier_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.pdtb_comparison_classifier_2.bias)

        self.pdtb_contingency_classifier_1 = nn.Linear(768 * 2, 768 * 2, bias=True)
        nn.init.xavier_uniform_(self.pdtb_contingency_classifier_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.pdtb_contingency_classifier_1.bias)

        self.pdtb_contingency_classifier_2 = nn.Linear(768 * 2, 2, bias=True)
        nn.init.xavier_uniform_(self.pdtb_contingency_classifier_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.pdtb_contingency_classifier_2.bias)


    def forward(self, all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label):
        all_cls_embed = self.sentence_words_embeddings(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask)[:,0,:]  # number of sentences in an article * cls embedding dim 768
        all_cls_embed = all_cls_embed.view(1, all_cls_embed.shape[0], all_cls_embed.shape[1])

        h0 = torch.zeros(2, 1, 384).cuda().requires_grad_()
        c0 = torch.zeros(2, 1, 384).cuda().requires_grad_()

        all_cls_embed_lstm, (_, _) = self.bilstm(all_cls_embed, (h0, c0)) # 1 * number of sentence in an article * 768

        final_sentences_features = all_cls_embed_lstm[0, :, :]

        info_bias_scores = self.info_bias_classifier_2(self.relu(self.info_bias_classifier_1(final_sentences_features)))  # number of sentences in an article * 2

        discourse_scores = self.discourse_classifier_2(self.relu(self.discourse_classifier_1(final_sentences_features)))  # number of sentences in an article * 9
        discourse_prob_predicted = self.softmax_1(discourse_scores)

        cross_entropy_loss = self.crossentropyloss(info_bias_scores, information_bias_label)
        mse_loss = self.mseloss(discourse_prob_predicted, discourse_prob_label)

        comparison_loss = torch.tensor(0.0, dtype=torch.float).to(device)
        contingency_loss = torch.tensor(0.0, dtype=torch.float).to(device)

        arg1_final_sentences_features = final_sentences_features[:-1,:]
        arg2_final_sentences_features = final_sentences_features[1:, :]

        comparison_predicted_score = self.pdtb_comparison_classifier_2(self.relu(self.pdtb_comparison_classifier_1(torch.cat((arg1_final_sentences_features, arg2_final_sentences_features), dim = 1))))
        comparison_loss += self.crossentropyloss(comparison_predicted_score, comp_prob_label)

        contingency_predicted_score = self.pdtb_contingency_classifier_2(self.relu(self.pdtb_contingency_classifier_1(torch.cat((arg1_final_sentences_features, arg2_final_sentences_features), dim = 1))))
        contingency_loss += self.crossentropyloss(contingency_predicted_score, cont_prob_label)

        return info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss









# stop here
