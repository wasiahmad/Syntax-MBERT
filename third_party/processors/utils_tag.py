# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
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
"""Utility functions for NER/POS tagging tasks."""

from __future__ import absolute_import, division, print_function

import logging
import os
import torch
import json

from io import open
from torch.utils.data import Dataset
from third_party.processors.tree import *
from third_party.processors.constants import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, langs=None,
                 heads=None, dep_tags=None, pos_tags=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs
        self.heads = heads
        self.dep_tags = dep_tags
        self.pos_tags = pos_tags


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            dep_tag_ids=None,
            pos_tag_ids=None,
            langs=None,
            root=None,
            heads=None,
            depths=None,
            trunc_token_ids=None,
            sep_token_indices=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.dep_tag_ids = dep_tag_ids
        self.pos_tag_ids = pos_tag_ids
        self.langs = langs
        self.root = root
        self.heads = heads
        self.trunc_token_ids = trunc_token_ids
        self.sep_token_indices = sep_token_indices
        self.depths = depths


def read_examples_from_file(file_path, lang, lang2id=None):
    guid_index = 1
    examples = []
    lang_id = lang2id.get(lang, lang2id["en"]) if lang2id else 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))

    if os.path.exists('{}.tsv'.format(file_path)):
        with open('{}.tsv'.format(file_path), encoding="utf-8") as f:
            words = []
            labels = []
            heads = []
            langs = []
            dep_tags = []
            pos_tags = []
            for line in f:
                line = line.strip()
                if not line:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(lang, guid_index),
                            words=words,
                            labels=labels,
                            langs=langs,
                            heads=heads,
                            dep_tags=dep_tags,
                            pos_tags=pos_tags
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    heads = []
                    dep_tags = []
                    pos_tags = []
                    continue

                splits = line.split("\t")
                words.append(splits[0])
                langs.append(lang_id)
                labels.append(splits[1])
                heads.append(int(splits[2]))
                dep_tags.append(splits[3].split(':')[0]
                                if ':' in splits[3] else splits[3])
                dep_tags.append(splits[4])

            if words:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(lang, guid_index),
                        words=words,
                        labels=labels,
                        langs=langs,
                        heads=heads,
                        dep_tags=dep_tags,
                        pos_tags=pos_tags
                    )
                )

    elif os.path.exists('{}.jsonl'.format(file_path)):
        with open('{}.jsonl'.format(file_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                ex = json.loads(line)
                examples.append(
                    InputExample(
                        guid="%s-%d".format(lang, guid_index),
                        words=ex['tokens'],
                        labels=ex['label'],
                        langs=[lang_id] * len(ex['tokens']),
                        heads=ex['head'],
                        dep_tags=[tag.split(':')[0] if ':' in tag else tag \
                                  for tag in ex['deptag']],
                        pos_tags=ex['postag'],
                    )
                )

    else:
        logger.info("[Warning] file {} with neither .tsv or .jsonl exists".format(file_path))
        return []

    return examples


def process_sentence(
        token_list, head_list, label_list, dep_tag_list,
        pos_tag_list, tokenizer, label_map, pad_token_label_id
):
    """
    When a token gets split into multiple word pieces,
    we make all the pieces (except the first) children of the first piece.
    However, only the first piece acts as the node that contains
    the dependent tokens as the children.
    """
    assert len(token_list) == len(head_list) == len(label_list) == \
           len(dep_tag_list) == len(pos_tag_list)

    text_tokens = []
    text_deptags = []
    text_postags = []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3  3    4  4
    sub_tok_to_orig_index = []
    # My name is Wa ##si Ah ##mad
    # 0  1    2  3       5
    old_index_to_new_index = []
    # My name is Wa ##si Ah ##mad
    # 1  1    1  1  0    1  0
    first_wpiece_indicator = []
    offset = 0
    labels = []
    for i, (token, label) in enumerate(zip(token_list, label_list)):
        word_tokens = tokenizer.tokenize(token)
        if len(token) != 0 and len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        old_index_to_new_index.append(offset)  # word piece index
        offset += len(word_tokens)
        for j, word_token in enumerate(word_tokens):
            first_wpiece_indicator += [1] if j == 0 else [0]
            labels += [label_map[label]] if j == 0 else [pad_token_label_id]
            text_tokens.append(word_token)
            sub_tok_to_orig_index.append(i)
            text_deptags.append(dep_tag_list[i])
            text_postags.append(pos_tag_list[i])

    assert len(text_tokens) == len(sub_tok_to_orig_index), \
        "{} != {}".format(len(text_tokens), len(sub_tok_to_orig_index))
    assert len(text_tokens) == len(first_wpiece_indicator)

    text_heads = []
    head_idx = -1
    assert max(head_list) <= len(head_list), (max(head_list), len(head_list))
    # iterating over the word pieces to adjust heads
    for i, orig_idx in enumerate(sub_tok_to_orig_index):
        # orig_idx: index of the original word (the word-piece belong to)
        head = head_list[orig_idx]
        if head == 0:  # root
            # if root word is split into multiple pieces,
            # we make the first piece as the root node
            # and all the other word pieces as the child of the root node
            if head_idx == -1:
                head_idx = i + 1
                text_heads.append(0)
            else:
                text_heads.append(head_idx)
        else:
            if first_wpiece_indicator[i] == 1:
                # head indices start from 1, so subtracting 1
                head = old_index_to_new_index[head - 1]
                text_heads.append(head + 1)
            else:
                # word-piece of a token (except the first)
                # so, we make the first piece the parent of all other word pieces
                head = old_index_to_new_index[orig_idx]
                text_heads.append(head + 1)

    assert len(text_tokens) == len(text_heads), \
        "{} != {}".format(len(text_tokens), len(text_heads))

    return text_tokens, text_heads, labels, text_deptags, text_postags


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        lang="en",
        use_syntax=False,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    special_tokens_count = 3 if sep_token_extra else 2

    features = []
    over_length_examples = 0
    wrong_examples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        if 0 not in example.heads:
            wrong_examples += 1
            continue

        tokens, heads, label_ids, dep_tags, pos_tags = process_sentence(
            example.words,
            example.heads,
            example.labels,
            example.dep_tags,
            example.pos_tags,
            tokenizer,
            label_map,
            pad_token_label_id
        )

        orig_text_len = len(tokens)
        root_idx = heads.index(0)
        text_offset = 1  # text_a follows <s>
        # So, we add 1 to head indices
        heads = np.add(heads, text_offset).tolist()
        # HEAD(<text_a> root) = index of <s> (1-based)
        heads[root_idx] = 1

        if len(tokens) > max_seq_length - special_tokens_count:
            # assert False  # we already truncated sequence
            # print("truncate token", len(tokens), max_seq_length, special_tokens_count)
            # tokens = tokens[: (max_seq_length - special_tokens_count)]
            # label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            over_length_examples += 1
            continue

        tokens += [tokenizer.sep_token]
        dep_tags += [tokenizer.sep_token]
        pos_tags += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [tokenizer.sep_token]
            dep_tags += [tokenizer.sep_token]
            pos_tags += [tokenizer.sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # cls_token_at_begining
        tokens = [tokenizer.cls_token] + tokens
        dep_tags = [tokenizer.cls_token] + dep_tags
        pos_tags = [tokenizer.cls_token] + pos_tags
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        if example.langs and len(example.langs) > 0:
            langs = [example.langs[0]] * max_seq_length
        else:
            print("example.langs", example.langs, example.words, len(example.langs))
            print("ex_index", ex_index, len(examples))
            langs = None

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(langs) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        #     logger.info("langs: {}".format(langs))

        one_ex_features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            langs=langs,
        )

        if use_syntax:
            #####################################################
            # prepare the UPOS and DEPENDENCY tag tensors
            #####################################################
            dep_tag_ids = deptag_to_id(dep_tags, tokenizer=str(type(tokenizer)))
            pos_tag_ids = upos_to_id(pos_tags, tokenizer=str(type(tokenizer)))

            if pad_on_left:
                dep_tag_ids = ([0] * padding_length) + dep_tag_ids
                pos_tag_ids = ([0] * padding_length) + pos_tag_ids
            else:
                dep_tag_ids += [0] * padding_length
                pos_tag_ids += [0] * padding_length

            assert len(input_ids) == len(dep_tag_ids)
            assert len(input_ids) == len(pos_tag_ids)
            assert len(dep_tag_ids) == max_seq_length
            assert len(pos_tag_ids) == max_seq_length

            one_ex_features.tag_ids = pos_tag_ids
            one_ex_features.dep_tag_ids = dep_tag_ids

            #####################################################
            # form the tree structure using head information
            #####################################################
            heads = [0] + heads + [1, 1] if sep_token_extra else [0] + heads + [1]
            assert len(tokens) == len(heads)
            root, nodes = head_to_tree(heads, tokens)
            assert len(heads) == root.size()
            sep_token_indices = [i for i, x in enumerate(tokens) if x == tokenizer.sep_token]
            depths = [nodes[i].depth() for i in range(len(nodes))]
            depths = np.asarray(depths, dtype=np.int32)

            one_ex_features.root = root
            one_ex_features.depths = depths
            one_ex_features.sep_token_indices = sep_token_indices

        features.append(one_ex_features)

    if over_length_examples > 0:
        logger.info('{} examples are discarded due to exceeding maximum length'.format(over_length_examples))
    if wrong_examples > 0:
        logger.info('{} wrong examples are discarded'.format(wrong_examples))
    return features


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


class SequenceDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Generates one sample of data"""
        feature = self.features[index]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        labels = torch.tensor(feature.label_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        dist_matrix = None
        depths = None
        dep_tag_ids = None
        pos_tag_ids = None
        if feature.root is not None:
            dep_tag_ids = torch.tensor(feature.dep_tag_ids, dtype=torch.long)
            pos_tag_ids = torch.tensor(feature.pos_tag_ids, dtype=torch.long)
            dist_matrix = root_to_dist_mat(feature.root)
            if feature.trunc_token_ids is not None:
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 0)  # delete rows
                dist_matrix = np.delete(dist_matrix, feature.trunc_token_ids, 1)  # delete columns

            dist_matrix = torch.tensor(dist_matrix, dtype=torch.long)  # seq_len x seq_len x max-path-len

        if feature.depths is not None:
            depths = feature.depths
            if feature.trunc_token_ids is not None:
                depths = np.delete(depths, feature.trunc_token_ids, 0)
            depths = torch.tensor(depths, dtype=torch.long)  # seq_len

        return [
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            dep_tag_ids,
            pos_tag_ids,
            dist_matrix,
            depths,
        ]


def batchify(batch):
    """Receives a batch of SequencePairDataset examples"""
    input_ids = torch.stack([data[0] for data in batch], dim=0)
    attention_mask = torch.stack([data[1] for data in batch], dim=0)
    token_type_ids = torch.stack([data[2] for data in batch], dim=0)
    labels = torch.stack([data[3] for data in batch], dim=0)

    dist_matrix = None
    depths = None
    dep_tag_ids = None
    pos_tag_ids = None

    if batch[0][4] is not None:
        dep_tag_ids = torch.stack([data[4] for data in batch], dim=0)

    if batch[0][5] is not None:
        pos_tag_ids = torch.stack([data[5] for data in batch], dim=0)

    if batch[0][6] is not None:
        dist_matrix = torch.full(
            (len(batch), input_ids.size(1), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen, slen = data[6].size()
            dist_matrix[i, :slen, :slen] = data[6]

    if batch[0][7] is not None:
        depths = torch.full(
            (len(batch), input_ids.size(1)), 99999, dtype=torch.long
        )
        for i, data in enumerate(batch):
            slen = data[7].size(0)
            depths[i, :slen] = data[7]

    return [
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        dep_tag_ids,
        pos_tag_ids,
        dist_matrix,
        depths
    ]
