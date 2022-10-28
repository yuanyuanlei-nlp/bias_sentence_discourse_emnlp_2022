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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup



train_data = pd.read_csv("./PDTB2/train_discourse.tsv", delimiter = '\t')
dev_data = pd.read_csv("./PDTB2/dev_discourse.tsv", delimiter = '\t')
test_data = pd.read_csv("./PDTB2/test_discourse.tsv", delimiter = '\t')


MAX_LEN = 128
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')



class custom_dataset(Dataset):
    def __init__(self, dataframe):
        self.custom_dataset = dataframe
        self.custom_dataset_arg1 = list(self.custom_dataset['Arg1'])
        self.custom_dataset_arg2 = list(self.custom_dataset['Arg2'])
        self.custom_dataset_comparison_label = list(self.custom_dataset['Comparison'])
        self.custom_dataset_contingency_label = list(self.custom_dataset['Contingency'])

    def __len__(self):
        return self.custom_dataset.shape[0]

    def __getitem__(self, idx):
        arg1_text = self.custom_dataset_arg1[idx]
        arg2_text = self.custom_dataset_arg2[idx]
        comparison_label = self.custom_dataset_comparison_label[idx]
        contingency_label = self.custom_dataset_contingency_label[idx]

        arg1_encoding = tokenizer.encode_plus(arg1_text, return_tensors='pt', add_special_tokens = True, max_length = MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = True, return_attention_mask = True)
        arg1_input_ids = arg1_encoding['input_ids'].view(MAX_LEN)
        arg1_token_type_ids = arg1_encoding['token_type_ids'].view(MAX_LEN)
        arg1_attention_mask = arg1_encoding['attention_mask'].view(MAX_LEN)

        arg2_encoding = tokenizer.encode_plus(arg2_text, return_tensors='pt', add_special_tokens = True, max_length = MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = True, return_attention_mask = True)
        arg2_input_ids = arg2_encoding['input_ids'].view(MAX_LEN)
        arg2_token_type_ids = arg2_encoding['token_type_ids'].view(MAX_LEN)
        arg2_attention_mask = arg2_encoding['attention_mask'].view(MAX_LEN)

        diction = {"arg1_input_ids": arg1_input_ids, "arg1_token_type_ids": arg1_token_type_ids, "arg1_attention_mask": arg1_attention_mask,
                   "arg2_input_ids": arg2_input_ids, "arg2_token_type_ids": arg2_token_type_ids, "arg2_attention_mask": arg2_attention_mask,
                   "comparison_label": comparison_label, "contingency_label": contingency_label}

        return diction



train_dataset = custom_dataset(train_data)
dev_dataset = custom_dataset(dev_data)
test_dataset = custom_dataset(test_data)




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





import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))



batch_size = 32
num_epochs = 10


no_decay = ['bias', 'LayerNorm.weight']
bert_weight_decay = 1e-2
non_bert_weight_decay = 1e-2
warmup_proportion = 0.1
bert_lr = 5e-5
non_bert_lr = 1e-4


comparison_model = Comparison_Model()
comparison_model.cuda()


param_all = list(comparison_model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': bert_weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'bert' in n)) ],  'lr': non_bert_lr, 'weight_decay': non_bert_weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('bert' in n)) ], 'lr': bert_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'bert' in n))], 'lr': non_bert_lr, 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, eps = 1e-8)
train_dataloader = list(DataLoader(train_dataset, batch_size=batch_size, shuffle = False))
num_train_steps = num_epochs * len(train_dataloader)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps)



seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



# comparison relation classifier training

best_macro_F = 0
best_comparison_F = 0

dev_dataloader = list(DataLoader(dev_dataset, batch_size=batch_size, shuffle=False))

for epoch_i in range(num_epochs):

    train_dataloader = list(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    num_batch = 0

    for i in range(len(train_dataloader)):

        if i % 100 == 0 and not i == 0:

            elapsed = format_time(time.time() - t0)
            avg_train_loss = total_train_loss / num_batch

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training Loss Average: {:.3f}'.format(i, len(train_dataloader), elapsed, avg_train_loss))

            total_train_loss = 0
            num_batch = 0

            # test on dev set
            comparison_model.eval()
            for j in range(len(dev_dataloader)):

                arg1_input_ids = dev_dataloader[j]['arg1_input_ids']
                arg1_token_type_ids = dev_dataloader[j]['arg1_token_type_ids']
                arg1_attention_mask = dev_dataloader[j]['arg1_attention_mask']
                arg2_input_ids = dev_dataloader[j]['arg2_input_ids']
                arg2_token_type_ids = dev_dataloader[j]['arg2_token_type_ids']
                arg2_attention_mask = dev_dataloader[j]['arg2_attention_mask']
                comparison_label = dev_dataloader[j]['comparison_label']
                contingency_label = dev_dataloader[j]['contingency_label']

                arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
                comparison_label, contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                                      arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                                      comparison_label.to(device), contingency_label.to(device)

                with torch.no_grad():
                    relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                                       arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                                       comparison_label, contingency_label)

                decision_supervised = torch.argmax(relation_score, dim=1).view(relation_score.shape[0], 1)
                true_label = comparison_label.view(relation_score.shape[0], 1)  # true label is comparison label

                if j == 0:
                    decision_supervised_onetest = decision_supervised
                    true_label_onetest = true_label
                else:
                    decision_supervised_onetest = torch.cat((decision_supervised_onetest, decision_supervised), dim=0)
                    true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

            decision_supervised_onetest = decision_supervised_onetest.to('cpu').numpy()
            true_label_onetest = true_label_onetest.to('cpu').numpy()

            macro_F = precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro')[2]
            comparison_F = precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary')[2]

            if macro_F > best_macro_F:
                torch.save(comparison_model.state_dict(), "./saved_models/comparison_best_macro_F.ckpt")
                best_macro_F = macro_F

            if comparison_F > best_comparison_F:
                torch.save(comparison_model.state_dict(), "./saved_models/comparison_best_comparison_F.ckpt")
                best_comparison_F = comparison_F

            print("MACRO: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro'))
            print("COMPARISON CLASS: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary'))



        comparison_model.train()
        arg1_input_ids = train_dataloader[i]['arg1_input_ids']
        arg1_token_type_ids = train_dataloader[i]['arg1_token_type_ids']
        arg1_attention_mask = train_dataloader[i]['arg1_attention_mask']
        arg2_input_ids = train_dataloader[i]['arg2_input_ids']
        arg2_token_type_ids = train_dataloader[i]['arg2_token_type_ids']
        arg2_attention_mask = train_dataloader[i]['arg2_attention_mask']
        comparison_label = train_dataloader[i]['comparison_label']
        contingency_label = train_dataloader[i]['contingency_label']

        arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
        comparison_label, contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                              arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                              comparison_label.to(device), contingency_label.to(device)

        optimizer.zero_grad()
        relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                           arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                           comparison_label, contingency_label)

        total_train_loss += supervised_loss.item()
        num_batch = num_batch + 1

        supervised_loss.backward()
        torch.nn.utils.clip_grad_norm_(comparison_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    training_time = format_time(time.time() - t0)
    print("")
    print("  Training epcoh took: {:}".format(training_time))



    # test on dev set
    comparison_model.eval()
    for j in range(len(dev_dataloader)):

        arg1_input_ids = dev_dataloader[j]['arg1_input_ids']
        arg1_token_type_ids = dev_dataloader[j]['arg1_token_type_ids']
        arg1_attention_mask = dev_dataloader[j]['arg1_attention_mask']
        arg2_input_ids = dev_dataloader[j]['arg2_input_ids']
        arg2_token_type_ids = dev_dataloader[j]['arg2_token_type_ids']
        arg2_attention_mask = dev_dataloader[j]['arg2_attention_mask']
        comparison_label = dev_dataloader[j]['comparison_label']
        contingency_label = dev_dataloader[j]['contingency_label']

        arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
        comparison_label, contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                              arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                              comparison_label.to(device), contingency_label.to(device)

        with torch.no_grad():
            relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                               arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                               comparison_label, contingency_label)

        decision_supervised = torch.argmax(relation_score, dim=1).view(relation_score.shape[0], 1)
        true_label = comparison_label.view(relation_score.shape[0], 1)  # true label is comparison label

        if j == 0:
            decision_supervised_onetest = decision_supervised
            true_label_onetest = true_label
        else:
            decision_supervised_onetest = torch.cat((decision_supervised_onetest, decision_supervised), dim=0)
            true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

    decision_supervised_onetest = decision_supervised_onetest.to('cpu').numpy()
    true_label_onetest = true_label_onetest.to('cpu').numpy()

    macro_F = precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro')[2]
    comparison_F = precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary')[2]

    if macro_F > best_macro_F:
        torch.save(comparison_model.state_dict(), "./saved_models/comparison_best_macro_F.ckpt")
        best_macro_F = macro_F

    if comparison_F > best_comparison_F:
        torch.save(comparison_model.state_dict(), "./saved_models/comparison_best_comparison_F.ckpt")
        best_comparison_F = comparison_F

    print("MACRO: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro'))
    print("COMPARISON CLASS: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary'))







# test on test set

# best macro F model

comparison_model = Comparison_Model()
comparison_model.cuda()

comparison_model.load_state_dict(torch.load('./saved_models/comparison_best_macro_F.ckpt', map_location=device))

test_dataloader = list(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

comparison_model.eval()

for j in range(len(test_dataloader)):

    arg1_input_ids = test_dataloader[j]['arg1_input_ids']
    arg1_token_type_ids = test_dataloader[j]['arg1_token_type_ids']
    arg1_attention_mask = test_dataloader[j]['arg1_attention_mask']
    arg2_input_ids = test_dataloader[j]['arg2_input_ids']
    arg2_token_type_ids = test_dataloader[j]['arg2_token_type_ids']
    arg2_attention_mask = test_dataloader[j]['arg2_attention_mask']
    comparison_label = test_dataloader[j]['comparison_label']
    contingency_label = test_dataloader[j]['contingency_label']

    arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
    comparison_label, contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                          arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                          comparison_label.to(device), contingency_label.to(device)

    with torch.no_grad():
        relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                           arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                           comparison_label, contingency_label)

    decision_supervised = torch.argmax(relation_score, dim=1).view(relation_score.shape[0], 1)
    true_label = comparison_label.view(relation_score.shape[0], 1)  # true label is comparison label

    if j == 0:
        decision_supervised_onetest = decision_supervised
        true_label_onetest = true_label
    else:
        decision_supervised_onetest = torch.cat((decision_supervised_onetest, decision_supervised), dim=0)
        true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

decision_supervised_onetest = decision_supervised_onetest.to('cpu').numpy()
true_label_onetest = true_label_onetest.to('cpu').numpy()

print("Best macro F model")
print("MACRO: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro'))
print("COMPARISON CLASS: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary'))



# best comparison class model

comparison_model = Comparison_Model()
comparison_model.cuda()

comparison_model.load_state_dict(torch.load('./saved_models/comparison_best_comparison_F.ckpt', map_location=device))

test_dataloader = list(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

comparison_model.eval()

for j in range(len(test_dataloader)):

    arg1_input_ids = test_dataloader[j]['arg1_input_ids']
    arg1_token_type_ids = test_dataloader[j]['arg1_token_type_ids']
    arg1_attention_mask = test_dataloader[j]['arg1_attention_mask']
    arg2_input_ids = test_dataloader[j]['arg2_input_ids']
    arg2_token_type_ids = test_dataloader[j]['arg2_token_type_ids']
    arg2_attention_mask = test_dataloader[j]['arg2_attention_mask']
    comparison_label = test_dataloader[j]['comparison_label']
    contingency_label = test_dataloader[j]['contingency_label']

    arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
    comparison_label, contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                          arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                          comparison_label.to(device), contingency_label.to(device)

    with torch.no_grad():
        relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                           arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                           comparison_label, contingency_label)

    decision_supervised = torch.argmax(relation_score, dim=1).view(relation_score.shape[0], 1)
    true_label = comparison_label.view(relation_score.shape[0], 1)  # true label is comparison label

    if j == 0:
        decision_supervised_onetest = decision_supervised
        true_label_onetest = true_label
    else:
        decision_supervised_onetest = torch.cat((decision_supervised_onetest, decision_supervised), dim=0)
        true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

decision_supervised_onetest = decision_supervised_onetest.to('cpu').numpy()
true_label_onetest = true_label_onetest.to('cpu').numpy()

print("Best comparison class F model")
print("MACRO: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='macro'))
print("COMPARISON CLASS: ", precision_recall_fscore_support(true_label_onetest, decision_supervised_onetest, average='binary'))







# predicted probability on BASIL

import json
softmax = nn.Softmax(dim = 1)

comparison_model = Comparison_Model()
comparison_model.cuda()

comparison_model.load_state_dict(torch.load('./saved_models/comparison_best_macro_F.ckpt', map_location=device))
comparison_model.eval()

def write_comp_prob(article_dir):

    with open(article_dir) as json_file:
        article_json = json.load(json_file)

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

    out_file_name = './BASIL_comp_probability/' + article_dir[8:-5] + '.txt'
    with open(out_file_name, 'a+') as outFile:
        outFile.write('sentence_index' + '\t' + 'prob_0' + '\t' + 'prob_1')
        for i in range(all_sentence_input_ids.shape[0] - 1):
            outFile.write('\n')
            outFile.write('S' + str(i + 1) + '\t')
            arg1_input_ids, arg1_token_type_ids, arg1_attention_mask = all_sentence_input_ids[i, :].view(1, MAX_LEN), all_sentence_token_type_ids[i, :].view(1, MAX_LEN), all_sentence_attention_mask[i, :].view(1, MAX_LEN)
            arg2_input_ids, arg2_token_type_ids, arg2_attention_mask = all_sentence_input_ids[(i + 1), :].view(1, MAX_LEN), all_sentence_token_type_ids[(i + 1), :].view(1, MAX_LEN), all_sentence_attention_mask[(i + 1), :].view(1, MAX_LEN)
            fake_comparison_label = torch.tensor(np.array([0] * 1), dtype=torch.long)
            fake_contingency_label = torch.tensor(np.array([0] * 1), dtype=torch.long)


            arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
            fake_comparison_label, fake_contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                                            arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                                            fake_comparison_label.to(device), fake_contingency_label.to(device)

            with torch.no_grad():
                relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                                   arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                                   fake_comparison_label, fake_contingency_label)

            prob = softmax(relation_score)
            prob = prob.cpu().numpy()

            prob_0 = prob[0,0]
            prob_1 = prob[0,1]
            outFile.write(str(prob_0) + '\t' + str(prob_1))


for triple_index in range(100):

    print(triple_index)

    article_hpo_dir = "./BASIL/" + str(triple_index) + '_hpo.json'
    article_fox_dir = "./BASIL/" + str(triple_index) + '_fox.json'
    article_nyt_dir = "./BASIL/" + str(triple_index) + '_nyt.json'

    write_comp_prob(article_hpo_dir)
    write_comp_prob(article_fox_dir)
    write_comp_prob(article_nyt_dir)







# predicted probability on BiasedSents

import json
softmax = nn.Softmax(dim = 1)

comparison_model = Comparison_Model()
comparison_model.cuda()

comparison_model.load_state_dict(torch.load('./saved_models/comparison_best_macro_F.ckpt', map_location=device))
comparison_model.eval()

def write_comp_prob(article_dir):

    with open(article_dir) as json_file:
        article_json = json.load(json_file)

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

    out_file_name = './BiasedSents_comp_probability/' + article_dir[14:-5] + '.txt'

    with open(out_file_name, 'a+') as outFile:
        outFile.write('sentence_index' + '\t' + 'prob_0' + '\t' + 'prob_1')

        for i in range(all_sentence_input_ids.shape[0] - 1):
            outFile.write('\n')
            outFile.write('S' + str(i + 1) + '\t')
            arg1_input_ids, arg1_token_type_ids, arg1_attention_mask = all_sentence_input_ids[i, :].view(1, MAX_LEN), all_sentence_token_type_ids[i, :].view(1, MAX_LEN), all_sentence_attention_mask[i, :].view(1, MAX_LEN)
            arg2_input_ids, arg2_token_type_ids, arg2_attention_mask = all_sentence_input_ids[(i + 1), :].view(1, MAX_LEN), all_sentence_token_type_ids[(i + 1), :].view(1, MAX_LEN), all_sentence_attention_mask[(i + 1), :].view(1, MAX_LEN)
            fake_comparison_label = torch.tensor(np.array([0] * 1), dtype=torch.long)
            fake_contingency_label = torch.tensor(np.array([0] * 1), dtype=torch.long)


            arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, \
            fake_comparison_label, fake_contingency_label = arg1_input_ids.to(device), arg1_token_type_ids.to(device), arg1_attention_mask.to(device), \
                                                            arg2_input_ids.to(device), arg2_token_type_ids.to(device), arg2_attention_mask.to(device), \
                                                            fake_comparison_label.to(device), fake_contingency_label.to(device)

            with torch.no_grad():
                relation_score, supervised_loss = comparison_model(arg1_input_ids, arg1_token_type_ids, arg1_attention_mask,
                                                                   arg2_input_ids, arg2_token_type_ids, arg2_attention_mask,
                                                                   fake_comparison_label, fake_contingency_label)

            prob = softmax(relation_score)
            prob = prob.cpu().numpy()

            prob_0 = prob[0,0]
            prob_1 = prob[0,1]
            outFile.write(str(prob_0) + '\t' + str(prob_1))


for triple_index in range(46):

    print(triple_index)

    article_dir = "./BiasedSents/" + str(triple_index) + '.json'

    write_comp_prob(article_dir)








# stop here
