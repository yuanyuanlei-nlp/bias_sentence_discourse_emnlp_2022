import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

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


folder0 = [42, 49, 0, 78, 81, 53, 82, 79, 47, 24]
folder1 = [76, 16, 18, 62, 14, 22, 35, 5, 45, 74]
folder2 = [6, 43, 61, 30, 83, 8, 27, 29, 99, 2]
folder3 = [37, 67, 71, 12, 68, 65, 57, 51, 77, 11]
folder4 = [19, 84, 54, 92, 88, 89, 72, 28, 34, 52]
folder5 = [17, 1, 59, 33, 21, 93, 38, 46, 86, 7]
folder6 = [3, 87, 40, 66, 4, 15, 25, 13, 60, 75]
folder7 = [31, 10, 50, 91, 98, 90, 55, 70, 96, 26]
folder8 = [48, 94, 41, 85, 20, 36, 64, 23, 9, 58]
folder9 = [73, 80, 44, 69, 97, 95, 32, 39, 56, 63]
folders = [folder0, folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8, folder9]




class Comparison_Sentence_Words_Embeddings(nn.Module):
    # input: input_ids, token_type_ids, attention_mask
    # output: batch_size * number of tokens * 768
    def __init__(self):
        super(Comparison_Sentence_Words_Embeddings, self).__init__()

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




class Comparison_Model(nn.Module):
    def __init__(self):
        super(Comparison_Model, self).__init__()

        self.sentence_words_embeddings = Comparison_Sentence_Words_Embeddings()

        self.relation_classifier = nn.Linear(768 * 2, 2, bias = True)
        nn.init.xavier_uniform_(self.relation_classifier.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.relation_classifier.bias)

        self.supervised_criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, comparison_label, contingency_label):

        arg1_words_embeddings = self.sentence_words_embeddings(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask) # batch_size * number of tokens * 768
        arg2_words_embeddings = self.sentence_words_embeddings(arg2_input_ids, arg2_token_type_ids, arg2_attention_mask) # batch_size * number of tokens * 768

        pair_embedding = torch.cat((arg1_words_embeddings[:, 0, :], arg2_words_embeddings[:, 0, :]), dim = 1) # batch_size * (768*2)
        relation_score = self.relation_classifier(pair_embedding) # batch_size * 2

        supervised_loss = self.supervised_criterion(relation_score, comparison_label)

        return relation_score, supervised_loss



from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

comparison_model = Comparison_Model()
comparison_model.cuda()

comparison_model.load_state_dict(torch.load('./saved_models/comparison_best_macro_F.ckpt', map_location=device))
comparison_model.eval()




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


        main_event_array = torch.argmax(discourse_prob_label, dim = 1).to('cpu').numpy()
        number_of_main_events = np.where(main_event_array == 1)[0].shape[0]

        info_bias_label_array = information_bias_label.to('cpu').numpy()
        number_of_bias = np.where(info_bias_label_array == 1)[0].shape[0]

        if ((number_of_main_events != 0) and (number_of_bias != 0)):

            main_events_input_ids = all_sentence_input_ids[np.where(main_event_array == 1)[0], :]
            main_events_token_type_ids = all_sentence_token_type_ids[np.where(main_event_array == 1)[0], :]
            main_events_attention_mask = all_sentence_attention_mask[np.where(main_event_array == 1)[0], :]
            main_events_final_sentences_features = final_sentences_features[np.where(main_event_array == 1)[0], :] # number of main events * 768

            for bias_i in range(number_of_bias):

                bias_index = np.where(info_bias_label_array == 1)[0][bias_i]

                bias_input_ids = all_sentence_input_ids[bias_index, :].repeat(number_of_main_events, 1)
                bias_token_type_ids = all_sentence_token_type_ids[bias_index, :].repeat(number_of_main_events, 1)
                bias_attention_mask = all_sentence_attention_mask[bias_index, :].repeat(number_of_main_events, 1)
                bias_final_sentences_features = final_sentences_features[bias_index, :].repeat(number_of_main_events, 1)

                fake_comparison_label = torch.tensor(np.array([0] * number_of_main_events), dtype=torch.long).to(device)
                fake_contingency_label = torch.tensor(np.array([0] * number_of_main_events), dtype=torch.long).to(device)

                comparison_score, _ = comparison_model(main_events_input_ids, main_events_token_type_ids, main_events_attention_mask,
                                                       bias_input_ids, bias_token_type_ids, bias_attention_mask, fake_comparison_label, fake_contingency_label)
                comparison_prob_label = self.softmax_1(comparison_score)

                comparison_predicted_score = self.pdtb_comparison_classifier_2(self.relu(self.pdtb_comparison_classifier_1(torch.cat((main_events_final_sentences_features, bias_final_sentences_features), dim=1))))
                comparison_loss += self.crossentropyloss(comparison_predicted_score, comparison_prob_label)

        return info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss









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








import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

no_decay = ['bias', 'LayerNorm.weight']
bert_weight_decay = 1e-2
non_bert_weight_decay = 1e-2
num_epochs = 5

lambda_news = 1.5
lambda_comp = 0.5
lambda_cont = 0.5

param_0 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_1 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_2 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_3 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_4 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_5 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_6 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_7 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_8 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
param_9 = {"warmup_proportion": , "non_bert_lr": , "bert_lr": }
parameters = [param_0, param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9]


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

info_bias_model = Joint_Classifier()
info_bias_model.cuda()

info_bias_model = Joint_Classifier()
info_bias_model.cuda()


for i in range(-1, 9):

    if i == -1:
        dev_folder_index = 9
        test_folder_index = 0
    else:
        dev_folder_index = i
        test_folder_index = i + 1


    para_dict = parameters[test_folder_index]
    warmup_proportion = para_dict['warmup_proportion']
    non_bert_lr = para_dict['non_bert_lr']
    bert_lr = para_dict['bert_lr']


    print("")
    print('======== Test Folder Index {:} ========'.format(test_folder_index))

    dev_triples = folders[dev_folder_index]
    test_triples = folders[test_folder_index]

    train_triples = []
    for j in range(10):
        if ((j != dev_folder_index) and (j != test_folder_index)):
            train_triples.extend(folders[j])

    train_triples = np.array(train_triples)  # not shuffled

    info_bias_model = Joint_Classifier()
    info_bias_model.cuda()

    param_all = list(info_bias_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': bert_weight_decay},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'bert' in n)) ],  'lr': non_bert_lr, 'weight_decay': non_bert_weight_decay},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'bert' in n))], 'lr': non_bert_lr, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps = 1e-8)
    num_train_steps = num_epochs * len(train_triples) * 3
    warmup_steps = int(warmup_proportion * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    best_biased_F = 0
    best_macro_F = 0

    for epoch_i in range(num_epochs):

        np.random.shuffle(train_triples)  # train_triples shuffled

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss_crossentropy = 0
        total_train_loss_mse = 0
        total_train_loss_comparison = 0
        total_train_loss_contingency = 0
        num_batch = 0

        for train_i in range(len(train_triples)):

            if train_i % 10 == 0 and not train_i == 0:

                elapsed = format_time(time.time() - t0)
                avg_train_loss_crossentropy = total_train_loss_crossentropy / num_batch
                avg_train_loss_mse = total_train_loss_mse / num_batch
                avg_train_loss_comparison = total_train_loss_comparison / num_batch
                avg_train_loss_contingency = total_train_loss_contingency / num_batch

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training cross entropy Loss Average: {:.3f}'.format(train_i, len(train_triples), elapsed, avg_train_loss_crossentropy))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training mse Loss Average: {:.3f}'.format(train_i, len(train_triples), elapsed, avg_train_loss_mse))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training comparison Loss Average: {:.3f}'.format(train_i, len(train_triples), elapsed, avg_train_loss_comparison))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training contingency Loss Average: {:.3f}'.format(train_i, len(train_triples), elapsed, avg_train_loss_contingency))

                total_train_loss_crossentropy = 0
                total_train_loss_mse = 0
                total_train_loss_comparison = 0
                total_train_loss_contingency = 0
                num_batch = 0

                # test on dev set
                info_bias_model.eval()
                for test_i in range(len(dev_triples)):

                    triple_index = dev_triples[test_i]

                    article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
                    article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
                    article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

                    article_discourse_hpo_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_hpo.txt'
                    article_discourse_fox_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_fox.txt'
                    article_discourse_nyt_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_nyt.txt'

                    article_comp_hpo_dir = "./BASIL_comp_probability/" + str(triple_index) + '_hpo.txt'
                    article_comp_fox_dir = "./BASIL_comp_probability/" + str(triple_index) + '_fox.txt'
                    article_comp_nyt_dir = "./BASIL_comp_probability/" + str(triple_index) + '_nyt.txt'

                    article_cont_hpo_dir = "./BASIL_cont_probability/" + str(triple_index) + '_hpo.txt'
                    article_cont_fox_dir = "./BASIL_cont_probability/" + str(triple_index) + '_fox.txt'
                    article_cont_nyt_dir = "./BASIL_cont_probability/" + str(triple_index) + '_nyt.txt'

                    article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_hpo_dir, article_discourse_hpo_dir, article_comp_hpo_dir, article_cont_hpo_dir  # hpo
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

                    with torch.no_grad():
                        info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

                    decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
                    true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
                    decision_triple = decision
                    true_label_triple = true_label

                    article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_fox_dir, article_discourse_fox_dir, article_comp_fox_dir, article_cont_fox_dir  # fox
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

                    with torch.no_grad():
                        info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

                    decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
                    true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
                    decision_triple = torch.cat((decision_triple, decision), dim=0)
                    true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

                    article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_nyt_dir, article_discourse_nyt_dir, article_comp_nyt_dir, article_cont_nyt_dir  # nyt
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
                    all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

                    with torch.no_grad():
                        info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

                    decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
                    true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
                    decision_triple = torch.cat((decision_triple, decision), dim=0)
                    true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

                    if test_i == 0:
                        decision_triple_onetest = decision_triple
                        true_label_triple_onetest = true_label_triple
                    else:
                        decision_triple_onetest = torch.cat((decision_triple_onetest, decision_triple), dim=0)
                        true_label_triple_onetest = torch.cat((true_label_triple_onetest, true_label_triple), dim=0)

                decision_triple_onetest = decision_triple_onetest.to('cpu').numpy()
                true_label_triple_onetest = true_label_triple_onetest.to('cpu').numpy()

                print("MACRO: ", precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest,average='macro'))
                print("BIASED: ", precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest,average='binary'))

                biased_F = precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='binary')[2]
                macro_F = precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='macro')[2]

                if biased_F > best_biased_F:
                    torch.save(info_bias_model.state_dict(), "./saved_models/joint_mse_comp_cont_best_biased_F_" + str(test_folder_index) + ".ckpt")
                    best_biased_F = biased_F

                if macro_F > best_macro_F:
                    torch.save(info_bias_model.state_dict(), "./saved_models/joint_mse_comp_cont_best_macro_F_" + str(test_folder_index) + ".ckpt")
                    best_macro_F = macro_F



            # train

            info_bias_model.train()

            triple_index = train_triples[train_i]

            article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
            article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
            article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

            article_discourse_hpo_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_hpo.txt'
            article_discourse_fox_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_fox.txt'
            article_discourse_nyt_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_nyt.txt'

            article_comp_hpo_dir = "./BASIL_comp_probability/" + str(triple_index) + '_hpo.txt'
            article_comp_fox_dir = "./BASIL_comp_probability/" + str(triple_index) + '_fox.txt'
            article_comp_nyt_dir = "./BASIL_comp_probability/" + str(triple_index) + '_nyt.txt'

            article_cont_hpo_dir = "./BASIL_cont_probability/" + str(triple_index) + '_hpo.txt'
            article_cont_fox_dir = "./BASIL_cont_probability/" + str(triple_index) + '_fox.txt'
            article_cont_nyt_dir = "./BASIL_cont_probability/" + str(triple_index) + '_nyt.txt'

            #batch_loss = 0
            optimizer.zero_grad()

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_hpo_dir, article_discourse_hpo_dir, article_comp_hpo_dir, article_cont_hpo_dir  # hpo
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)
            batch_loss = cross_entropy_loss + lambda_news * mse_loss + lambda_comp * comparison_loss + lambda_cont * contingency_loss
            total_train_loss_crossentropy += cross_entropy_loss.item()
            total_train_loss_mse += mse_loss.item()
            total_train_loss_comparison += comparison_loss.item()
            total_train_loss_contingency += contingency_loss.item()

            num_batch = num_batch + 1
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(info_bias_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_fox_dir, article_discourse_fox_dir, article_comp_fox_dir, article_cont_fox_dir  # fox
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)
            batch_loss = cross_entropy_loss + lambda_news * mse_loss + lambda_comp * comparison_loss + lambda_cont * contingency_loss
            total_train_loss_crossentropy += cross_entropy_loss.item()
            total_train_loss_mse += mse_loss.item()
            total_train_loss_comparison += comparison_loss.item()
            total_train_loss_contingency += contingency_loss.item()

            num_batch = num_batch + 1
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(info_bias_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_nyt_dir, article_discourse_nyt_dir, article_comp_nyt_dir, article_cont_nyt_dir  # nyt
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)
            batch_loss = cross_entropy_loss + lambda_news * mse_loss + lambda_comp * comparison_loss + lambda_cont * contingency_loss
            total_train_loss_crossentropy += cross_entropy_loss.item()
            total_train_loss_mse += mse_loss.item()
            total_train_loss_comparison += comparison_loss.item()
            total_train_loss_contingency += contingency_loss.item()

            num_batch = num_batch + 1
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(info_bias_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()



        elapsed = format_time(time.time() - t0)
        avg_train_loss_crossentropy = total_train_loss_crossentropy / num_batch
        avg_train_loss_mse = total_train_loss_mse / num_batch
        avg_train_loss_comparison = total_train_loss_comparison / num_batch
        avg_train_loss_contingency = total_train_loss_contingency / num_batch

        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training cross entropy Loss Average: {:.3f}'.format(train_i, len(train_triples), elapsed, avg_train_loss_crossentropy))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training mse Loss Average: {:.3f}'.format(train_i,len(train_triples),elapsed,avg_train_loss_mse))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training comparison Loss Average: {:.3f}'.format(train_i,len(train_triples),elapsed,avg_train_loss_comparison))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training contingency Loss Average: {:.3f}'.format(train_i,len(train_triples),elapsed,avg_train_loss_contingency))

        total_train_loss_crossentropy = 0
        total_train_loss_mse = 0
        total_train_loss_comparison = 0
        total_train_loss_contingency = 0
        num_batch = 0

        # test on dev set
        info_bias_model.eval()
        for test_i in range(len(dev_triples)):

            triple_index = dev_triples[test_i]

            article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
            article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
            article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

            article_discourse_hpo_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_hpo.txt'
            article_discourse_fox_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_fox.txt'
            article_discourse_nyt_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_nyt.txt'

            article_comp_hpo_dir = "./BASIL_comp_probability/" + str(triple_index) + '_hpo.txt'
            article_comp_fox_dir = "./BASIL_comp_probability/" + str(triple_index) + '_fox.txt'
            article_comp_nyt_dir = "./BASIL_comp_probability/" + str(triple_index) + '_nyt.txt'

            article_cont_hpo_dir = "./BASIL_cont_probability/" + str(triple_index) + '_hpo.txt'
            article_cont_fox_dir = "./BASIL_cont_probability/" + str(triple_index) + '_fox.txt'
            article_cont_nyt_dir = "./BASIL_cont_probability/" + str(triple_index) + '_nyt.txt'

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_hpo_dir, article_discourse_hpo_dir, article_comp_hpo_dir, article_cont_hpo_dir  # hpo
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            with torch.no_grad():
                info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

            decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
            true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
            decision_triple = decision
            true_label_triple = true_label

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_fox_dir, article_discourse_fox_dir, article_comp_fox_dir, article_cont_fox_dir  # fox
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            with torch.no_grad():
                info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

            decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
            true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
            decision_triple = torch.cat((decision_triple, decision), dim=0)
            true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

            article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_nyt_dir, article_discourse_nyt_dir, article_comp_nyt_dir, article_cont_nyt_dir  # nyt
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
            all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

            with torch.no_grad():
                info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

            decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
            true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
            decision_triple = torch.cat((decision_triple, decision), dim=0)
            true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

            if test_i == 0:
                decision_triple_onetest = decision_triple
                true_label_triple_onetest = true_label_triple
            else:
                decision_triple_onetest = torch.cat((decision_triple_onetest, decision_triple), dim=0)
                true_label_triple_onetest = torch.cat((true_label_triple_onetest, true_label_triple), dim=0)

        decision_triple_onetest = decision_triple_onetest.to('cpu').numpy()
        true_label_triple_onetest = true_label_triple_onetest.to('cpu').numpy()

        print("MACRO: ", precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='macro'))
        print("BIASED: ", precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='binary'))

        biased_F = precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='binary')[2]
        macro_F = precision_recall_fscore_support(true_label_triple_onetest, decision_triple_onetest, average='macro')[2]

        if biased_F > best_biased_F:
            torch.save(info_bias_model.state_dict(), "./saved_models/joint_mse_comp_cont_best_biased_F_" + str(test_folder_index) + ".ckpt")
            best_biased_F = biased_F

        if macro_F > best_macro_F:
            torch.save(info_bias_model.state_dict(), "./saved_models/joint_mse_comp_cont_best_macro_F_" + str(test_folder_index) + ".ckpt")
            best_macro_F = macro_F



        training_time = format_time(time.time() - t0)
        print("")
        print("  Training epcoh took: {:}".format(training_time))



    # best biased model test on test set

    print("")
    print("======== Testing on test folder: {:} ========".format(test_folder_index))

    print("Best Biased F is: {:}".format(best_biased_F))

    info_bias_model = Joint_Classifier()
    info_bias_model.cuda()

    info_bias_model.load_state_dict(torch.load("./saved_models/joint_mse_comp_cont_best_biased_F_" + str(test_folder_index) + ".ckpt", map_location=device))

    info_bias_model.eval()
    for test_i in range(len(test_triples)):

        triple_index = test_triples[test_i]

        article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
        article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
        article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

        article_discourse_hpo_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_hpo.txt'
        article_discourse_fox_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_fox.txt'
        article_discourse_nyt_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_nyt.txt'

        article_comp_hpo_dir = "./BASIL_comp_probability/" + str(triple_index) + '_hpo.txt'
        article_comp_fox_dir = "./BASIL_comp_probability/" + str(triple_index) + '_fox.txt'
        article_comp_nyt_dir = "./BASIL_comp_probability/" + str(triple_index) + '_nyt.txt'

        article_cont_hpo_dir = "./BASIL_cont_probability/" + str(triple_index) + '_hpo.txt'
        article_cont_fox_dir = "./BASIL_cont_probability/" + str(triple_index) + '_fox.txt'
        article_cont_nyt_dir = "./BASIL_cont_probability/" + str(triple_index) + '_nyt.txt'

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_hpo_dir, article_discourse_hpo_dir, article_comp_hpo_dir, article_cont_hpo_dir  # hpo
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = decision
        true_label_triple = true_label

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_fox_dir, article_discourse_fox_dir, article_comp_fox_dir, article_cont_fox_dir  # fox
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = torch.cat((decision_triple, decision), dim=0)
        true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_nyt_dir, article_discourse_nyt_dir, article_comp_nyt_dir, article_cont_nyt_dir  # nyt
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = torch.cat((decision_triple, decision), dim=0)
        true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

        if test_i == 0:
            decision_triple_onetest_biased = decision_triple
            true_label_triple_onetest_biased = true_label_triple
        else:
            decision_triple_onetest_biased = torch.cat((decision_triple_onetest_biased, decision_triple), dim=0)
            true_label_triple_onetest_biased = torch.cat((true_label_triple_onetest_biased, true_label_triple), dim=0)


    decision_triple_onetest_biased = decision_triple_onetest_biased.to('cpu').numpy()
    true_label_triple_onetest_biased = true_label_triple_onetest_biased.to('cpu').numpy()
    print("MACRO: ", precision_recall_fscore_support(true_label_triple_onetest_biased, decision_triple_onetest_biased, average='macro'))
    print("BIASED: ", precision_recall_fscore_support(true_label_triple_onetest_biased, decision_triple_onetest_biased, average='binary'))
    print("======== Finished Testing on test folder: {:} ========".format(test_folder_index))


    # best macro model test on test set

    print("")
    print("======== Testing on test folder: {:} ========".format(test_folder_index))

    print("Best Macro F is: {:}".format(best_macro_F))

    info_bias_model = Joint_Classifier()
    info_bias_model.cuda()

    info_bias_model.load_state_dict(torch.load("./saved_models/joint_mse_comp_cont_best_macro_F_" + str(test_folder_index) + ".ckpt", map_location=device))

    info_bias_model.eval()
    for test_i in range(len(test_triples)):

        triple_index = test_triples[test_i]

        article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
        article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
        article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

        article_discourse_hpo_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_hpo.txt'
        article_discourse_fox_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_fox.txt'
        article_discourse_nyt_dir = "./BASIL_discourse_probability/" + str(triple_index) + '_nyt.txt'

        article_comp_hpo_dir = "./BASIL_comp_probability/" + str(triple_index) + '_hpo.txt'
        article_comp_fox_dir = "./BASIL_comp_probability/" + str(triple_index) + '_fox.txt'
        article_comp_nyt_dir = "./BASIL_comp_probability/" + str(triple_index) + '_nyt.txt'

        article_cont_hpo_dir = "./BASIL_cont_probability/" + str(triple_index) + '_hpo.txt'
        article_cont_fox_dir = "./BASIL_cont_probability/" + str(triple_index) + '_fox.txt'
        article_cont_nyt_dir = "./BASIL_cont_probability/" + str(triple_index) + '_nyt.txt'

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_hpo_dir, article_discourse_hpo_dir, article_comp_hpo_dir, article_cont_hpo_dir  # hpo
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = decision
        true_label_triple = true_label

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_fox_dir, article_discourse_fox_dir, article_comp_fox_dir, article_cont_fox_dir  # fox
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = torch.cat((decision_triple, decision), dim=0)
        true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

        article_dir, article_discourse_dir, article_comp_dir, article_cont_dir = article_nyt_dir, article_discourse_nyt_dir, article_comp_nyt_dir, article_cont_nyt_dir  # nyt
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = AllSentences_InfoBiasLabels_DiscourseProbs_in_Article(article_dir, article_discourse_dir, article_comp_dir, article_cont_dir)
        all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label = all_sentence_input_ids.to(device), all_sentence_token_type_ids.to(device), all_sentence_attention_mask.to(device), information_bias_label.to(device), discourse_prob_label.to(device), comp_prob_label.to(device), cont_prob_label.to(device)

        with torch.no_grad():
            info_bias_scores, discourse_prob_predicted, cross_entropy_loss, mse_loss, comparison_loss, contingency_loss = info_bias_model(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask, information_bias_label, discourse_prob_label, comp_prob_label, cont_prob_label)

        decision = torch.argmax(info_bias_scores, dim=1).view(info_bias_scores.shape[0], 1)
        true_label = information_bias_label.view(info_bias_scores.shape[0], 1)
        decision_triple = torch.cat((decision_triple, decision), dim=0)
        true_label_triple = torch.cat((true_label_triple, true_label), dim=0)

        if test_i == 0:
            decision_triple_onetest_macro = decision_triple
            true_label_triple_onetest_macro = true_label_triple
        else:
            decision_triple_onetest_macro = torch.cat((decision_triple_onetest_macro, decision_triple), dim=0)
            true_label_triple_onetest_macro = torch.cat((true_label_triple_onetest_macro, true_label_triple), dim=0)


    decision_triple_onetest_macro = decision_triple_onetest_macro.to('cpu').numpy()
    true_label_triple_onetest_macro = true_label_triple_onetest_macro.to('cpu').numpy()
    print("MACRO: ", precision_recall_fscore_support(true_label_triple_onetest_macro, decision_triple_onetest_macro, average='macro'))
    print("BIASED: ", precision_recall_fscore_support(true_label_triple_onetest_macro, decision_triple_onetest_macro, average='binary'))
    print("======== Finished Testing on test folder: {:} ========".format(test_folder_index))


    if test_folder_index == 0:
        decision_triple_alltests_biased = decision_triple_onetest_biased
        true_label_triple_alltests_biased = true_label_triple_onetest_biased
    else:
        decision_triple_alltests_biased = np.concatenate((decision_triple_alltests_biased, decision_triple_onetest_biased), axis = 0)
        true_label_triple_alltests_biased = np.concatenate((true_label_triple_alltests_biased, true_label_triple_onetest_biased), axis = 0)


    if test_folder_index == 0:
        decision_triple_alltests_macro = decision_triple_onetest_macro
        true_label_triple_alltests_macro = true_label_triple_onetest_macro
    else:
        decision_triple_alltests_macro = np.concatenate((decision_triple_alltests_macro, decision_triple_onetest_macro), axis = 0)
        true_label_triple_alltests_macro = np.concatenate((true_label_triple_alltests_macro, true_label_triple_onetest_macro), axis = 0)


    if test_folder_index == 0:
        if precision_recall_fscore_support(true_label_triple_onetest_macro, decision_triple_onetest_macro, average='binary')[2] > precision_recall_fscore_support(true_label_triple_onetest_biased, decision_triple_onetest_biased, average='binary')[2]:
            decision_triple_alltests_fusion = decision_triple_onetest_macro
            true_label_triple_alltests_fusion = true_label_triple_onetest_macro
        else:
            decision_triple_alltests_fusion = decision_triple_onetest_biased
            true_label_triple_alltests_fusion = true_label_triple_onetest_biased
    else:
        if precision_recall_fscore_support(true_label_triple_onetest_macro, decision_triple_onetest_macro, average='binary')[2] > precision_recall_fscore_support(true_label_triple_onetest_biased, decision_triple_onetest_biased, average='binary')[2]:
            decision_triple_alltests_fusion = np.concatenate((decision_triple_alltests_fusion, decision_triple_onetest_macro), axis = 0)
            true_label_triple_alltests_fusion = np.concatenate((true_label_triple_alltests_fusion, true_label_triple_onetest_macro), axis=0)
        else:
            decision_triple_alltests_fusion = np.concatenate((decision_triple_alltests_fusion, decision_triple_onetest_biased), axis=0)
            true_label_triple_alltests_fusion = np.concatenate((true_label_triple_alltests_fusion, true_label_triple_onetest_biased), axis=0)


print("======== Results on all test folders ========")
print("MACRO: ", precision_recall_fscore_support(true_label_triple_alltests_fusion, decision_triple_alltests_fusion, average='macro'))
print("BIASED: ", precision_recall_fscore_support(true_label_triple_alltests_fusion, decision_triple_alltests_fusion, average='binary'))
print("Classification Report \n", classification_report(true_label_triple_alltests_fusion, decision_triple_alltests_fusion, digits = 4))
print("Confusion Matrix \n", confusion_matrix(true_label_triple_alltests_fusion, decision_triple_alltests_fusion))






















# stop here
