import os
import copy
import json
import torch
import logging

from transformers import DataProcessor
from torch.utils.data import Dataset
from third_party.processors.tree import *
from third_party.processors.constants import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
      sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
      Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
      specified for train and dev examples, but not for test examples.
    """

    def __init__(
            self,
            guid,
            text_a,
            text_b=None,
            label=None,
            language=None,
            text_a_heads=None,
            text_b_heads=None,
            text_a_upos=None,
            text_b_upos=None,
            text_a_deptags=None,
            text_b_deptags=None,
    ):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_a_heads = text_a_heads
        self.text_b_heads = text_b_heads
        self.text_a_upos = text_a_upos
        self.text_b_upos = text_b_upos
        self.text_a_deptags = text_a_deptags
        self.text_b_deptags = text_b_deptags
        self.label = label
        self.language = language

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
      input_ids: Indices of input sequence tokens in the vocabulary.
      attention_mask: Mask to avoid performing attention on padding token indices.
        Mask values selected in ``[0, 1]``:
        Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
      token_type_ids: Segment token indices to indicate first and second portions of the inputs.
      label: Label corresponding to the input
    """

    def __init__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            tag_ids=None,
            dep_tag_ids=None,
            position_ids=None,
            langs=None,
            label=None,
            root=None,
            trunc_token_ids=None,
            text_a_b_tokens=None,
            text_a_b_heads=None,
            text_a_b_deptags=None,
            depths=None,
            sep_token_indices=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.tag_ids = tag_ids
        self.dep_tag_ids = dep_tag_ids
        self.label = label
        self.langs = langs
        self.root = root
        self.trunc_token_ids = trunc_token_ids
        self.text_a_b_tokens = text_a_b_tokens
        self.text_a_b_heads = text_a_b_heads
        self.text_a_b_deptags = text_a_b_deptags
        self.depths = depths
        self.sep_token_indices = sep_token_indices

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def process_sentence(token_list, head_list, upos_list, deptags, tokenizer):
    """
    When a token gets split into multiple word pieces,
    we make all the pieces (except the first) children of the first piece.
    However, only the first piece acts as the node that contains
    the dependent tokens as the children.
    """
    assert len(token_list) == len(head_list) == len(upos_list) == len(deptags)

    text_tokens = []
    text_upos = []
    text_deptags = []
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
    for i, token in enumerate(token_list):
        sub_tokens = tokenizer.tokenize(token)
        if len(sub_tokens) == 0:
            sub_tokens = [tokenizer.unk_token]
        old_index_to_new_index.append(offset)  # word piece index
        offset += len(sub_tokens)
        for j, sub_token in enumerate(sub_tokens):
            first_wpiece_indicator += [1] if j == 0 else [0]
            text_tokens.append(sub_token)
            sub_tok_to_orig_index.append(i)
            text_upos.append(upos_list[i])
            text_deptags.append(deptags[i])

    assert len(text_tokens) == len(sub_tok_to_orig_index), \
        "{} != {}".format(len(text_tokens), len(sub_tok_to_orig_index))
    assert len(text_tokens) == len(first_wpiece_indicator)

    text_heads = []
    head_idx = -1
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

    return text_tokens, text_heads, text_upos, text_deptags


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        output_mode=None,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        lang2id=None,
        use_syntax=False,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
      examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
      tokenizer: Instance of a tokenizer that will tokenize the examples
      max_length: Maximum example length
      task: GLUE task
      label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
      output_mode: String indicating the output mode. Either ``regression`` or ``classification``
      pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
      pad_token: Padding token
      pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
      mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
        and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
        actual values)
    Returns:
      If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
      containing the task-specific features. If the input is a list of ``InputExamples``, will return
      a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    special_tokens_count = 3 if sep_token_extra else 2

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        text_a_tokens, text_a_heads, text_a_upos, text_a_deptags = process_sentence(
            example.text_a,
            example.text_a_heads,
            example.text_a_upos,
            example.text_a_deptags,
            tokenizer,
        )
        orig_text_a_len = len(text_a_tokens)
        a_root_idx = text_a_heads.index(0)
        text_a_offset = 1  # text_a follows <s>
        # So, we add 1 to head indices
        text_a_heads = np.add(text_a_heads, text_a_offset).tolist()
        # HEAD(<text_a> root) = index of <s> (1-based)
        text_a_heads[a_root_idx] = 1

        text_b_tokens, text_b_heads, text_b_upos, text_b_deptags = process_sentence(
            example.text_b,
            example.text_b_heads,
            example.text_b_upos,
            example.text_b_deptags,
            tokenizer,
        )
        orig_text_b_len = len(text_b_tokens)
        # point the root of text_b to special start token: <s>
        b_root_idx = text_b_heads.index(0)
        # text_b follows <s> text_a </s> </s>
        # so we add 3 to orig_text_a_len
        text_b_offset = special_tokens_count + orig_text_a_len
        text_b_heads = np.add(text_b_heads, text_b_offset).tolist()
        # HEAD(<text_b> root) = index of <s> (1-based)
        text_b_heads[b_root_idx] = 1

        #####################################################
        # lets perform the truncation following LONGEST_FIRST
        #####################################################
        total_tokens = orig_text_a_len + orig_text_b_len + special_tokens_count + 1
        num_tokens_to_remove = total_tokens - max_length
        trunc_text_a_tokens = list(text_a_tokens)
        trunc_text_b_tokens = list(text_b_tokens)
        trunc_text_a_upos = list(text_a_upos)
        trunc_text_b_upos = list(text_b_upos)
        trunc_text_a_deptags = list(text_a_deptags)
        trunc_text_b_deptags = list(text_b_deptags)

        for _ in range(num_tokens_to_remove):
            if len(trunc_text_a_tokens) > len(trunc_text_b_tokens):
                trunc_text_a_tokens = trunc_text_a_tokens[:-1]
                trunc_text_a_upos = trunc_text_a_upos[:-1]
                trunc_text_a_deptags = trunc_text_a_deptags[:-1]
            else:
                trunc_text_b_tokens = trunc_text_b_tokens[:-1]
                trunc_text_b_upos = trunc_text_b_upos[:-1]
                trunc_text_b_deptags = trunc_text_b_deptags[:-1]

        #####################################################
        # Tokenize, prepare other tensors
        #####################################################
        text_a_token_ids = tokenizer.convert_tokens_to_ids(trunc_text_a_tokens)
        text_b_token_ids = tokenizer.convert_tokens_to_ids(trunc_text_b_tokens)

        input_ids = tokenizer.build_inputs_with_special_tokens(
            text_a_token_ids, text_b_token_ids
        )
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(
            text_a_token_ids, text_b_token_ids
        )

        input_length = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - input_length
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        lid = lang2id.get(example.language, lang2id["en"]) if lang2id is not None else 0
        langs = [lid] * max_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
                len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
                len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 3:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[:input_length]]))
        #     logger.info(
        #         "sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids[:input_length]))
        #     )
        #     logger.info(
        #         "attention_mask: %s" % " ".join([str(x) for x in attention_mask[:input_length]])
        #     )
        #     logger.info(
        #         "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids[:input_length]])
        #     )
        #     logger.info("pos_tag_ids: %s" % " ".join([str(x) for x in pos_tag_ids[:input_length]]))
        #     logger.info("dep_tag_ids: %s" % " ".join([str(x) for x in dep_tag_ids[:input_length]]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))
        #     logger.info("language: %s, (lid = %d)" % (example.language, lid))

        one_ex_features = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            langs=langs,
            label=label,
        )

        if use_syntax:
            #####################################################
            # prepare the UPOS and DEPENDENCY tag tensors
            #####################################################
            text_a_b_upos = [tokenizer.cls_token] + trunc_text_a_upos + [tokenizer.sep_token]
            text_a_b_deptags = [tokenizer.cls_token] + trunc_text_a_deptags + [tokenizer.sep_token]
            if sep_token_extra:
                text_a_b_upos += [tokenizer.sep_token]
                text_a_b_deptags += [tokenizer.sep_token]

            text_a_b_upos += trunc_text_b_upos + [tokenizer.sep_token]
            text_a_b_deptags += trunc_text_b_deptags + [tokenizer.sep_token]

            pos_tag_ids = upos_to_id(text_a_b_upos, tokenizer=str(type(tokenizer)))
            # assert len(input_ids) == len(pos_tag_ids) # to early to assert 
            dep_tag_ids = deptag_to_id(text_a_b_deptags, tokenizer=str(type(tokenizer)))
            # assert len(input_ids) == len(dep_tag_ids) # to early to assert

            if pad_on_left:
                pos_tag_ids = ([0] * padding_length) + pos_tag_ids
                dep_tag_ids = ([0] * padding_length) + dep_tag_ids
            else:
                pos_tag_ids = pos_tag_ids + ([0] * padding_length)
                dep_tag_ids = dep_tag_ids + ([0] * padding_length)

            assert len(input_ids) == len(pos_tag_ids) # you should assert after padding
            assert len(input_ids) == len(dep_tag_ids) # you should assert after padding
            assert len(dep_tag_ids) == max_length
            assert len(pos_tag_ids) == max_length
            one_ex_features.tag_ids = pos_tag_ids
            one_ex_features.dep_tag_ids = dep_tag_ids

            #####################################################
            # form the tree structure using head information
            #####################################################
            text_heads = [0] + text_a_heads + [len(text_a_tokens) + 1]
            sep_token_indices = [0] * (orig_text_a_len + 1) + [1]
            if sep_token_extra:
                text_heads += [len(text_a_tokens) + 1]
                sep_token_indices += [1]
            text_heads += text_b_heads + [total_tokens - 1]
            sep_token_indices += [0] * orig_text_b_len + [1]

            root, nodes = head_to_tree(text_heads)
            assert len(nodes) == root.size()
            depths = [nodes[i].depth() for i in range(len(nodes))]
            depths = np.asarray(depths, dtype=np.int32)

            one_ex_features.root = root
            one_ex_features.depths = depths
            one_ex_features.sep_token_indices = sep_token_indices

            #####################################################
            # store truncated token info to modify tree structure
            #####################################################
            trunc_token_ids = []
            if len(trunc_text_a_tokens) < orig_text_a_len:  # text_a is truncated
                text_a_idx_to_delete = list(range(len(trunc_text_a_tokens), orig_text_a_len))
                text_a_idx_to_delete = np.add(
                    text_a_idx_to_delete, text_a_offset
                ).tolist()
                trunc_token_ids.extend(text_a_idx_to_delete)
            if len(trunc_text_b_tokens) < orig_text_b_len:  # context is truncated
                text_b_idx_to_delete = list(range(len(trunc_text_b_tokens), orig_text_b_len))
                text_b_idx_to_delete = np.add(
                    text_b_idx_to_delete, text_b_offset
                ).tolist()
                trunc_token_ids.extend(text_b_idx_to_delete)
            one_ex_features.trunc_token_ids = trunc_token_ids

        features.append(one_ex_features)

    return features


class SequencePairDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Generates one sample of data"""
        feature = self.features[index]
        # Convert to Tensors and build dataset
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        labels = torch.tensor(feature.label, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)

        dist_matrix = None
        depths = None
        dep_tag_ids = None
        pos_tag_ids = None
        if feature.root is not None:
            dep_tag_ids = torch.tensor(feature.dep_tag_ids, dtype=torch.long)
            pos_tag_ids = torch.tensor(feature.tag_ids, dtype=torch.long)
            dist_matrix = root_to_dist_mat(feature.root)
            if feature.trunc_token_ids:
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


class PawsxProcessor(DataProcessor):
    """Processor for the PAWS-X dataset."""

    def __init__(self):
        pass

    def get_mix_examples(self, data_dir, language, split='test'):
        examples = []
        languages = language.split('_')

        filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, languages[0]))
        with open(filename, encoding='utf-8') as f:
            pre_lines = [json.loads(l.rstrip("\n")) for l in f]
        filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, languages[1]))
        with open(filename, encoding='utf-8') as f:
            hyp_lines = [json.loads(l.rstrip("\n")) for l in f]

        assert len(pre_lines) == len(hyp_lines)

        ex_discarded = 0
        for i, (pline, hline) in enumerate(zip(pre_lines, hyp_lines)):
            guid = "%s-%s-%s" % (split, language, i)
            premise = pline['premise']
            hypothesis = hline['hypothesis']
            if pline['label'] != hline['label']:
                ex_discarded += 1
                continue

            text_a = premise['tokens']
            text_a_heads = premise['head']
            text_a_upos = premise['upos']
            text_a_deptags = premise['deprel']
            text_a_deptags = [tag.split(':')[0] if ':' in tag else tag \
                              for tag in text_a_deptags]

            text_b = hypothesis['tokens']
            text_b_heads = hypothesis['head']
            text_b_upos = hypothesis['upos']
            text_b_deptags = hypothesis['deprel']
            text_b_deptags = [tag.split(':')[0] if ':' in tag else tag \
                              for tag in text_b_deptags]

            label = str(pline['label'].strip())

            assert isinstance(text_a, list) and isinstance(text_b, list) and isinstance(label, str)
            assert isinstance(text_a_heads, list) and isinstance(text_b_heads, list)
            assert isinstance(text_a_upos, list) and isinstance(text_b_upos, list)
            assert isinstance(text_a_deptags, list) and isinstance(text_b_deptags, list)

            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label, language=language,
                text_a_heads=text_a_heads, text_b_heads=text_b_heads,
                text_a_upos=text_a_upos, text_b_upos=text_b_upos,
                text_a_deptags=text_a_deptags, text_b_deptags=text_b_deptags
            ))

        logger.warning('{} examples discarded...'.format(ex_discarded))
        return examples

    def get_examples(self, data_dir, language='en', split='train', swap_pairs=False):
        """See base class."""
        examples = []
        for lg in language.split(','):
            filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, lg))
            with open(filename, encoding='utf-8') as f:
                lines = [json.loads(l.rstrip("\n")) for l in f]

            # lines = lines[:10000]  # used in debugging
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, lg, i)
                if swap_pairs:
                    premise = line['hypothesis']
                    hypothesis = line['premise']
                else:
                    premise = line['premise']
                    hypothesis = line['hypothesis']

                text_a = premise['tokens']
                text_a_heads = premise['head']
                text_a_upos = premise['upos']
                text_a_deptags = premise['deprel']
                text_a_deptags = [tag.split(':')[0] if ':' in tag else tag \
                                  for tag in text_a_deptags]

                text_b = hypothesis['tokens']
                text_b_heads = hypothesis['head']
                text_b_upos = hypothesis['upos']
                text_b_deptags = hypothesis['deprel']
                text_b_deptags = [tag.split(':')[0] if ':' in tag else tag \
                                  for tag in text_b_deptags]

                label = str(line['label'].strip())

                assert isinstance(text_a, list) and isinstance(text_b, list) and isinstance(label, str)
                assert isinstance(text_a_heads, list) and isinstance(text_b_heads, list)
                assert isinstance(text_a_upos, list) and isinstance(text_b_upos, list)
                assert isinstance(text_a_deptags, list) and isinstance(text_b_deptags, list)

                examples.append(InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg,
                    text_a_heads=text_a_heads, text_b_heads=text_b_heads,
                    text_a_upos=text_a_upos, text_b_upos=text_b_upos,
                    text_a_deptags=text_a_deptags, text_b_deptags=text_b_deptags
                ))

        return examples

    def get_train_examples(self, data_dir, language="en", swap_pairs=False):
        """See base class."""
        return self.get_examples(data_dir, language, split="train", swap_pairs=swap_pairs)

    def get_translate_train_examples(self, data_dir, language="en"):
        """See base class."""
        raise NotImplementedError

    def get_translate_test_examples(self, data_dir, language="en"):
        """See base class."""
        raise NotImplementedError

    def get_test_examples(self, data_dir, language="en", swap_pairs=False):
        """See base class."""
        if '_' in language:
            return self.get_mix_examples(data_dir, language, split='test')
        else:
            return self.get_examples(data_dir, language, split='test', swap_pairs=swap_pairs)

    def get_dev_examples(self, data_dir, language="en"):
        """See base class."""
        return self.get_examples(data_dir, language, split="dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


pawsx_processors = {
    "pawsx": PawsxProcessor,
}

pawsx_output_modes = {
    "pawsx": "classification",
}

pawsx_tasks_num_labels = {
    "pawsx": 2,
}


class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset."""

    def __init__(self):
        pass

    def get_mix_examples(self, data_dir, language, split='test'):
        examples = []
        languages = language.split('_')

        filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, languages[0]))
        with open(filename, encoding='utf-8') as f:
            pre_lines = [json.loads(l.rstrip("\n")) for l in f]
        filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, languages[1]))
        with open(filename, encoding='utf-8') as f:
            hyp_lines = [json.loads(l.rstrip("\n")) for l in f]

        assert len(pre_lines) == len(hyp_lines)

        ex_discarded = 0
        for i, (pline, hline) in enumerate(zip(pre_lines, hyp_lines)):
            guid = "%s-%s-%s" % (split, language, i)
            premise = pline['premise']
            hypothesis = hline['hypothesis']
            if pline['label'] != hline['label']:
                ex_discarded += 1
                continue

            text_a = premise['tokens']
            text_a_heads = premise['head']
            text_a_upos = premise['upos']
            text_a_deptags = premise['deprel']
            text_a_deptags = [tag.split(':')[0] if ':' in tag else tag \
                              for tag in text_a_deptags]

            text_b = hypothesis['tokens']
            text_b_heads = hypothesis['head']
            text_b_upos = hypothesis['upos']
            text_b_deptags = hypothesis['deprel']
            text_b_deptags = [tag.split(':')[0] if ':' in tag else tag \
                              for tag in text_b_deptags]

            label = str(pline['label'].strip())

            assert isinstance(text_a, list) and isinstance(text_b, list) and isinstance(label, str)
            assert isinstance(text_a_heads, list) and isinstance(text_b_heads, list)
            assert isinstance(text_a_upos, list) and isinstance(text_b_upos, list)
            assert isinstance(text_a_deptags, list) and isinstance(text_b_deptags, list)

            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label, language=language,
                text_a_heads=text_a_heads, text_b_heads=text_b_heads,
                text_a_upos=text_a_upos, text_b_upos=text_b_upos,
                text_a_deptags=text_a_deptags, text_b_deptags=text_b_deptags
            ))

        logger.warning('{} examples discarded...'.format(ex_discarded))
        return examples

    def get_examples(self, data_dir, language='en', split='train', swap_pairs=False):
        """See base class."""
        examples = []
        for lg in language.split(','):
            filename = os.path.join(data_dir, "{}-{}.jsonl".format(split, lg))
            with open(filename, encoding='utf-8') as f:
                lines = [json.loads(l.rstrip("\n")) for l in f]

            # lines = lines[:10000]  # used in debugging
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, lg, i)
                if swap_pairs:
                    premise = line['hypothesis']
                    hypothesis = line['premise']
                else:
                    premise = line['premise']
                    hypothesis = line['hypothesis']

                text_a = premise['tokens']
                text_a_heads = premise['head']
                text_a_upos = premise['upos']
                text_a_deptags = premise['deprel']
                text_a_deptags = [tag.split(':')[0] if ':' in tag else tag \
                                  for tag in text_a_deptags]

                text_b = hypothesis['tokens']
                text_b_heads = hypothesis['head']
                text_b_upos = hypothesis['upos']
                text_b_deptags = hypothesis['deprel']
                text_b_deptags = [tag.split(':')[0] if ':' in tag else tag \
                                  for tag in text_b_deptags]

                label = str(line['label'].strip())

                assert isinstance(text_a, list) and isinstance(text_b, list) and isinstance(label, str)
                assert isinstance(text_a_heads, list) and isinstance(text_b_heads, list)
                assert isinstance(text_a_upos, list) and isinstance(text_b_upos, list)
                assert isinstance(text_a_deptags, list) and isinstance(text_b_deptags, list)

                examples.append(InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg,
                    text_a_heads=text_a_heads, text_b_heads=text_b_heads,
                    text_a_upos=text_a_upos, text_b_upos=text_b_upos,
                    text_a_deptags=text_a_deptags, text_b_deptags=text_b_deptags
                ))

        return examples

    def get_train_examples(self, data_dir, language='en', swap_pairs=False):
        return self.get_examples(data_dir, language, split='train', swap_pairs=swap_pairs)

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en', swap_pairs=False):
        if '_' in language:
            return self.get_mix_examples(data_dir, language, split='test')
        else:
            return self.get_examples(data_dir, language, split='test', swap_pairs=swap_pairs)

    def get_translate_train_examples(self, data_dir, language='en'):
        """See base class."""
        raise NotImplementedError

    def get_translate_test_examples(self, data_dir, language='en'):
        raise NotImplementedError

    def get_pseudo_test_examples(self, data_dir, language='en'):
        raise NotImplementedError

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


xnli_processors = {
    "xnli": XnliProcessor,
}

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}
