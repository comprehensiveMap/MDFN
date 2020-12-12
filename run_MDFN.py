# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
import datetime
import argparse
import csv
import logging
import os
import random
import sys
import pickle
import numpy as np
import torch
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import glob

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (BertConfig, BertForMultipleChoice, BertTokenizer,
                            ElectraConfig, ElectraTokenizer, RobertaConfig, RobertaTokenizer, RobertaForMultipleChoice)
from modeling import (ElectraForMultipleChoicePlus, Baseline, BertBaseline, RobertaBaseline, BertForMultipleChoicePlus, RobertaForMultipleChoicePlus)
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
import re
import os

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoicePlus, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoicePlus, RobertaTokenizer),
    'electra': (ElectraConfig, ElectraForMultipleChoicePlus, ElectraTokenizer)
}

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'sep_pos': sep_pos,
                'turn_ids': turn_ids
            }
            for input_ids, input_mask, segment_ids, sep_pos, turn_ids in choices_features
        ]
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class DoubanProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = [line["r"]]
            text_b = [line["m"].strip().split("[SEP]")]
            label = line["y"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            label_list = []
            message_list = []
            response_list = []
            label_any_1 = 0
            for ids,line in enumerate(f):
                line = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f\\x7f]').sub(' ', line).strip()
                line = line.strip().replace("_", "")
                parts = line.strip().split("\t")
                lable = parts[0]
                message = ""
                for i in range(1, len(parts) - 1, 1):
                    part = parts[i].strip()
                    if len(part) > 0:
                        message += part
                        message += " [SEP] "
                response = parts[-1]
                
                if lable == '1':
                    label_any_1 = 1
                label_list.append(lable)
                message_list.append(message)
                response_list.append(response)
                if ids % 10 == 9:
                    if label_any_1 == 1:
                        for lable,message,response in zip(label_list,message_list,response_list):
                            data = {"y": lable, "m": message, "r": response}
                            lines.append(data)
                    label_any_1 = 0
                    label_list = []
                    message_list = []
                    response_list = []
            return lines

class UbuntuProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_data(self, input_file):
        """Reads a tab separated value file."""

        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i,line in enumerate(f):
                line = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f\\x7f]').sub(' ', line).strip()
                line = line.strip().replace("_", "")
                parts = line.strip().split("\t")
                lable = parts[0]
                message = ""
                for i in range(1, len(parts) - 1, 1):
                    part = parts[i].strip()
                    if len(part) > 0:
                        if i != len(parts) - 2:
                            message += part
                            message += "[SEP]"
                        else:
                            message += part
                response = parts[-1]
                data = {"y": lable, "m": message, "r": response}
                lines.append(data)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = [line["r"]]
            text_b = [line["m"].strip().split("[SEP]")]
            label = line["y"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MuTualProcessor(DataProcessor):
    """Processor for the MuTual data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, 'train')
        file = self._read_txt(file)
        return self._create_examples(file, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        file = os.path.join(data_dir, 'dev')
        file = self._read_txt(file)
        return self._create_examples(file, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, 'test')
        file = self._read_txt(file)
        return self._create_examples(file, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            id = "%s-%s" % (set_type, data_raw["id"])
            article = data_raw["article"]

            article = re.split(r"(f : |m : |M: |F: )", article)
            article = ["".join(i) for i in zip(article[1::2], article[2::2])]

            truth = str(ord(data_raw['answers']) - ord('A'))
            options = data_raw['options']

            examples.append(
                InputExample(
                    guid=id,
                    text_a = [options[0], options[1], options[2], options[3]],
                    text_b=[article, article, article, article], # this is not efficient but convenient
                    label=truth))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, max_utterance_num,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        choices_features = []
        all_tokens = []

        for ending_idx, (text_a, text_b) in enumerate(zip(example.text_a, example.text_b)):
            tokens_a = tokenizer.tokenize(text_a)

            tokens_b = []

            for idx, text in enumerate(text_b):
                if len(text.strip()) > 0:
                    tokens_b.extend(tokenizer.tokenize(text) + ["[SEP]"])
            
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)

            tokens = ["[CLS]"]
            turn_ids = [0]

            context_len = []
            sep_pos = []
            
            tokens_b_raw = " ".join(tokens_b)
            tokens_b = []
            current_pos = 0
            for toks in tokens_b_raw.split("[SEP]")[-max_utterance_num - 1:-1]:
                context_len.append(len(toks.split()) + 1)
                tokens_b.extend(toks.split())
                tokens_b.extend(["[SEP]"])
                current_pos += context_len[-1]
                turn_ids += [len(sep_pos)] * context_len[-1]
                sep_pos.append(current_pos)
                
            tokens += tokens_b

            segment_ids = [0] * (len(tokens))

            tokens_a += ["[SEP]"]
            tokens += tokens_a
            segment_ids += [1] * (len(tokens_a))
            
            turn_ids += [len(sep_pos)] * len(tokens_a) 
            sep_pos.append(len(tokens) - 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            turn_ids += padding

            context_len += [-1] * (max_utterance_num - len(context_len))
            sep_pos += [0] * (max_utterance_num + 1 - len(sep_pos))

            assert len(sep_pos) == max_utterance_num + 1
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(context_len) == max_utterance_num 
            assert len(turn_ids) == max_seq_length 

            choices_features.append((input_ids, input_mask, segment_ids, sep_pos, turn_ids))
            all_tokens.append(tokens)


        label_id = label_map[example.label]
        

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            for choice_idx, (input_ids, input_mask, segment_ids, sep_pos, turn_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("tokens: %s" % " ".join([str(x) for x in all_tokens[choice_idx]]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("sep_pos: %s" % " ".join([str(x) for x in sep_pos]))
                logger.info("turn_ids: %s" % " ".join([str(x) for x in turn_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                example_id = example.guid, 
                choices_features = choices_features,
                label = label_id
                )
        )

    return features
            

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop(0)

def get_p_at_n_in_m(pred, n, m, ind):
    pos_score = pred[ind]
    curr = pred[ind:ind + m]
    curr = sorted(curr, reverse=True)

    if curr[n - 1] <= pos_score:
        return 1
    return 0

def mean_average_precision(sort_data):
    # to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index + 1)
    return sum_precision / count_1


def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0


def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    
    return 1.0 * select_lable.count(1) / sort_lable.count(1)

def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5

def evaluate_douban(pred, label):
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    total_num = 0
    data = []
    #print(label)
    for i in range(0, len(label)):
        if i % 10 == 0:
            data = []
        data.append((float(pred[i]), int(label[i])))
        if i % 10 == 9:
            total_num += 1
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5
    # print('total num: %s' %total_num)
    # print('MAP: %s' %(1.0*sum_m_a_p/total_num))
    # print('MRR: %s' %(1.0*sum_m_r_r/total_num))
    # print('P@1: %s' %(1.0*sum_p_1/total_num))
    return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
            1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)

def evaluate(pred, label):
    # assert len(data) % 10 == 0

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = int(len(pred) / 10)

    for i in range(0, length):
        ind = i * 10
        assert label[ind] == 1

        p_at_1_in_2 += get_p_at_n_in_m(pred, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(pred, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(pred, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(pred, 5, 10, ind)

    return (p_at_1_in_2 / length, p_at_1_in_10 / length, p_at_2_in_10 / length, p_at_5_in_10 / length)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def ComputeR10(scores,labels,count = 10):
    total = 0
    correct1 = 0
    correct5 = 0
    correct2 = 0
    correct10 = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if np.argmax(sublist) < 1:
                correct1 = correct1 + 1
            if np.argmax(sublist) < 2:
                correct2 = correct2 + 1
            if np.argmax(sublist) < 5:
                correct5 = correct5 + 1
            if np.argmax(sublist) < 10:
                correct10 = correct10 + 1
    print(correct1, correct5, correct10, total)
    return (float(correct1)/ total, float(correct2)/ total, float(correct5)/ total, float(correct10)/ total)

def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    return (float(correct)/ total)

def Compute_R4_2(preds, labels):
    p2 = 0
    for i in range(len(preds)):
        j = sorted(list(preds[i]), reverse = True)
        if j.index(preds[i][labels[i]]) <= 1:
            p2 += 1
    return p2 / len(preds)

def Compute_MRR(preds, labels):
    mrr = 0
    for i in range(len(preds)):
        j = sorted(list(preds[i]), reverse = True)
        mrr += 1 / (j.index(preds[i][labels[i]]) + 1)
    
    return mrr / len(preds)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if preds.shape[1] == 1:
        preds_class = np.ones(preds.shape)
        preds_class[preds < 0] = 0
    else:
        preds_class = np.argmax(preds, axis=1)
    preds_logits = preds[:, 1]

    if task_name in ["ubuntu", "ecd"]:
        return {"acc": simple_accuracy(preds_class, labels), "recall@10":ComputeR10(preds_logits, labels), "recall@2":ComputeR2_1(preds_logits, labels), "DAM":evaluate(preds_logits, labels)}
    elif task_name == 'douban':
        return {"DAM":evaluate_douban(preds_logits, labels)}
    elif task_name in ['mutual']:
        return {"R4_1": simple_accuracy(preds_class, labels), "R4_2": Compute_R4_2(preds, labels), "MRR:": Compute_MRR(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='../../../MuTual/data/mutual',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default="google/electra-large-discriminator", type=str)
    parser.add_argument("--model_type", default="electra", type = str,
                        help = "Pre-trained Model selected in the list: bert, roberta, electra")
    parser.add_argument("--task_name",
                        default="mutual",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="output_mutual_electra_3",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_utterance_num",
                        default=20,
                        type=int,
                        help="The maximum total utterance number.")
    parser.add_argument("--cache_flag",
                        default="v1",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--max_grad_norm", 
                        default = 1.0, 
                        type = float,
                        help = "The maximum grad norm for clipping")
    parser.add_argument("--cache_dir",
                        default='../../cached_models',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--baseline",
                        action='store_true',
                        help="Whether to run baseline.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=4e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_rnn",
                        default=1,
                        type=int,
                        help="RNN.")
    parser.add_argument("--num_decouple",
                        default=1,
                        type=int,
                        help="Decoupling Layers.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "ubuntu": UbuntuProcessor,
        'douban': DoubanProcessor,
        'ecd': UbuntuProcessor,
        "mutual": MuTualProcessor
    }

    output_modes = {
        "ubuntu": "classification",
        "mutual": "classification",
        'douban': "classification",
        'ecd': 'classification'
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.baseline:
        if args.model_type == 'electra':
            model_class = Baseline
        elif args.model_type == 'bert':
            model_class = BertBaseline
        elif args.model_type == 'roberta':
            model_class = RobertaBaseline

    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name, 
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.model_name_or_path.split('/'))).pop(), "train",str(args.task_name), str(args.max_seq_length),
            str(args.max_utterance_num), str(args.cache_flag))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer, output_mode)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        # (batch_size, 1, seq_len)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        #all_response_len = torch.tensor(select_field(train_features, 'response_len'), dtype=torch.long)
        all_sep_pos = torch.tensor(select_field(train_features, 'sep_pos'), dtype=torch.long)
        all_turn_ids = torch.tensor(select_field(train_features, 'turn_ids'), dtype = torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_pos, all_turn_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = processor.get_dev_examples(args.data_dir)
        cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.model_name_or_path.split('/'))).pop(), "valid",str(args.task_name), str(args.max_seq_length),
            str(args.max_utterance_num), str(args.cache_flag))
        eval_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer, output_mode)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving eval features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_sep_pos = torch.tensor(select_field(eval_features, 'sep_pos'), dtype=torch.long)
        all_turn_ids = torch.tensor(select_field(eval_features, 'turn_ids'), dtype = torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_pos, all_turn_ids, all_label_ids)
        # Run prediction for full data

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            #nb_tr_examples = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                          'sep_pos': batch[3],
                          'turn_ids': batch[4],
                          'labels': batch[5]}
                
                #input_ids, input_mask, segment_ids, response_len, sep_pos, label_ids = batch

                output = model(**inputs)
                loss = output[0]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.detach().item()
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1

            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(args.output_dir, str(epoch) + "_" + WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = None

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                          'sep_pos': batch[3],
                          'turn_ids': batch[4],
                          'labels': batch[5]}
                    #outputs = eval_model(**inputs)
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.detach().mean().item()
                
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            
            result = compute_metrics(task_name, preds, out_label_ids)
            loss = tr_loss / nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
