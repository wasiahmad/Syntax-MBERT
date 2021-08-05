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
""" Finetuning multi-lingual models on XNLI/PAWSX (Bert, XLM, XLMRoberta)."""

import argparse
import glob
import logging
import os
import h5py
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from third_party.modeling_bert import BertForSequenceClassification
from third_party.processors.constants import *

from third_party.processors.utils_classify import (
    convert_examples_to_features,
    SequencePairDataset,
    batchify,
    XnliProcessor,
    PawsxProcessor
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig,)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}

PROCESSORS = {
    "xnli": XnliProcessor,
    "pawsx": PawsxProcessor,
}


def compute_metrics(preds, labels):
    scores = {
        "acc": (preds == labels).mean(),
        "num": len(preds),
        "correct": (preds == labels).sum(),
    }
    return scores


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(
        args,
        model,
        tokenizer,
        split="train",
        language="en",
        lang2id=None,
        prefix="",
        output_file=None,
        label_list=None,
        output_only_prediction=True,
):
    """Evalute the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args,
            eval_task,
            tokenizer,
            split=split,
            language=language,
            lang2id=lang2id,
            evaluate=True,
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=batchify,
        )

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} {} *****".format(prefix, language))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sentences = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) if t is not None else None for t in batch)

            with torch.no_grad():
                inputs = dict()
                inputs['input_ids'] = batch[0]
                inputs['attention_mask'] = batch[1]
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert"] else None
                inputs['labels'] = batch[3]

                if args.use_syntax:
                    inputs["dep_tag_ids"] = batch[4]
                    inputs["pos_tag_ids"] = batch[5]
                    inputs["dist_mat"] = batch[6]
                    inputs["tree_depths"] = batch[7]

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                sentences = inputs["input_ids"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                sentences = np.append(
                    sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        if output_file:
            logger.info("***** Save prediction ******")
            with open(output_file, "w") as fout:
                pad_token_id = tokenizer.pad_token_id
                sentences = sentences.astype(int).tolist()
                sentences = [[w for w in s if w != pad_token_id] for s in sentences]
                sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
                # fout.write('Prediction\tLabel\tSentences\n')
                for p, l, s in zip(list(preds), list(out_label_ids), sentences):
                    s = " ".join(s)
                    if label_list:
                        p = label_list[p]
                        l = label_list[l]
                    if output_only_prediction:
                        fout.write(str(p) + "\n")
                    else:
                        fout.write("{}\t{}\t{}\n".format(p, l, s))
        logger.info("***** Eval results {} {} *****".format(prefix, language))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results


def load_and_cache_examples(
        args, task, tokenizer, split="train", language="en", lang2id=None, evaluate=False
):
    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = PROCESSORS[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    lc = "_lc" if args.do_lower_case else ""
    cached_features_file = os.path.join(
        args.output_dir,
        "cached_{}_{}_{}_{}_{}{}".format(
            split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            language,
            lc,
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.data_dir, language, args.swap_pairs)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            lang2id=lang2id,
            use_syntax=args.use_syntax,
        )

        # NOTE. WE do not cache the features as we will do this experiment less often
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if args.model_type == "xlm":
        raise NotImplementedError
    else:
        dataset = SequencePairDataset(features)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
             + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--train_language",
        default="en",
        type=str,
        help="Train language if is different of the evaluation language.",
    )
    parser.add_argument(
        "--predict_languages",
        type=str,
        default="en",
        help="prediction languages separated by ','.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task_name",
        default="xnli",
        type=str,
        required=True,
        help="The task name",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run prediction."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--test_split", type=str, default="test", help="split of training set"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--log_file", default="train", type=str, help="log file")
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    #
    #
    parser.add_argument(
        "--syntactic_layers",
        type=str, default='0,1,2',
        help="comma separated layer indices for syntax fusion",
    )
    parser.add_argument(
        "--num_syntactic_heads",
        default=2, type=int,
        help="Number of syntactic heads",
    )
    parser.add_argument(
        "--use_syntax",
        type='bool',
        default=False,
        help="Whether to use syntax-based modeling",
    )
    parser.add_argument(
        "--use_dependency_tag",
        type='bool',
        default=False,
        help="Whether to use dependency tag in structure modeling",
    )
    parser.add_argument(
        "--use_pos_tag",
        type='bool',
        default=False,
        help="Whether to use pos tags in structure modeling",
    )
    parser.add_argument(
        "--use_structural_loss",
        type='bool',
        default=False,
        help="Whether to use structural loss along with task loss",
    )
    parser.add_argument(
        "--struct_loss_coeff",
        default=1.0, type=float,
        help="Multiplying factor for the structural loss",
    )
    parser.add_argument(
        "--max_syntactic_distance",
        default=1, type=int,
        help="Max distance to consider during graph attention",
    )
    parser.add_argument(
        "--num_gat_layer",
        default=4, type=int,
        help="Number of layers in Graph Attention Networks (GAT)",
    )
    parser.add_argument(
        "--num_gat_head",
        default=4, type=int,
        help="Number of attention heads in Graph Attention Networks (GAT)",
    )
    parser.add_argument(
        "--batch_normalize",
        action="store_true",
        help="Apply batch normalization to <s> representation",
    )
    parser.add_argument(
        "--swap_pairs",
        action="store_true",
        help="Swap the input sentence pairs.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, args.log_file)),
            logging.StreamHandler(),
        ],
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.info("Input args: %r" % args)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare dataset
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    ####################################
    config.dep_tag_vocab_size = len(DEPTAG_SYMBOLS) + NUM_SPECIAL_TOKENS
    config.pos_tag_vocab_size = len(POS_SYMBOLS) + NUM_SPECIAL_TOKENS
    config.use_dependency_tag = args.use_dependency_tag
    config.use_pos_tag = args.use_pos_tag
    config.use_structural_loss = args.use_structural_loss
    config.struct_loss_coeff = args.struct_loss_coeff
    config.num_syntactic_heads = args.num_syntactic_heads
    config.syntactic_layers = args.syntactic_layers
    config.max_syntactic_distance = args.max_syntactic_distance
    config.use_syntax = args.use_syntax
    config.batch_normalize = args.batch_normalize
    config.num_gat_layer = args.num_gat_layer
    config.num_gat_head = args.num_gat_head
    ####################################

    logger.info("config = {}".format(config))

    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)

    if os.path.exists(os.path.join(args.output_dir, "checkpoint-best")):
        best_checkpoint = os.path.join(args.output_dir, "checkpoint-best")
    else:
        best_checkpoint = args.output_dir

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path if args.model_name_or_path else best_checkpoint,
            do_lower_case=args.do_lower_case,
        )
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)
        output_predict_file = os.path.join(
            args.output_dir, "xling_" + args.test_split + "_results.txt"
        )
        total = total_correct = 0.0
        with open(output_predict_file, "a") as writer:
            writer.write(
                "======= Predict using the model from {} for {}:\n".format(
                    best_checkpoint, args.test_split
                )
            )
            for pre_lang in args.predict_languages.split(","):
                for hyp_lang in args.predict_languages.split(","):
                    if pre_lang == hyp_lang:
                        continue
                    # output_file = os.path.join(
                    #     args.output_dir, "test-{}.tsv".format(language)
                    # )
                    language = pre_lang + '_' + hyp_lang
                    result = evaluate(
                        args,
                        model,
                        tokenizer,
                        split=args.test_split,
                        language=language,
                        lang2id=lang2id,
                        prefix="best_checkpoint",
                        # output_file=output_file,
                        label_list=label_list,
                    )
                    logger.info("{}={}".format(language, result["acc"]))
                    writer.write("=====================\nlanguage={}\n".format(language))
                    for key in sorted(result.keys()):
                        writer.write("{} = {}\n".format(key, result[key]))
                    total += result["num"]
                    total_correct += result["correct"]

            writer.write("=====================\n")
            writer.write("total={}\n".format(total_correct / total))

    return result


if __name__ == "__main__":
    main()
