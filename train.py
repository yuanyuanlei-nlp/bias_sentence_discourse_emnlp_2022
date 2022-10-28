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
warmup_proportion = 0.1
non_bert_lr = 1e-4
bert_lr = 2e-5
lambda_news = 1.5
lambda_comp = 0.5
lambda_cont = 0.5

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


print("======== Results on all test folders ========")

print("======== Best Biased F Model ========")
print("MACRO: ", precision_recall_fscore_support(true_label_triple_alltests_biased, decision_triple_alltests_biased, average='macro'))
print("BIASED: ", precision_recall_fscore_support(true_label_triple_alltests_biased, decision_triple_alltests_biased, average='binary'))
print("Classification Report \n", classification_report(true_label_triple_alltests_biased, decision_triple_alltests_biased, digits = 4))
print("Confusion Matrix \n", confusion_matrix(true_label_triple_alltests_biased, decision_triple_alltests_biased))

print("======== Best Macro F Model ========")
print("MACRO: ", precision_recall_fscore_support(true_label_triple_alltests_macro, decision_triple_alltests_macro, average='macro'))
print("BIASED: ", precision_recall_fscore_support(true_label_triple_alltests_macro, decision_triple_alltests_macro, average='binary'))
print("Classification Report \n", classification_report(true_label_triple_alltests_macro, decision_triple_alltests_macro, digits = 4))
print("Confusion Matrix \n", confusion_matrix(true_label_triple_alltests_macro, decision_triple_alltests_macro))









# stop here
