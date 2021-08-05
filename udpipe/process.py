import sys

sys.path.append(".")
sys.path.append("..")

import json
import argparse
from tqdm import tqdm
from conllu import parse
from conllify import Model
from multiprocessing import Pool
from collections import OrderedDict
from multiprocessing.util import Finalize
from third_party.processors.tree import head_to_tree

TOK = None
LANGUAGES = None
UDPIPE_MODELS = None


def init():
    global TOK, LANGUAGES
    TOK = UDPipeTokenizer(LANGUAGES, UDPIPE_MODELS)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


class UDPipeTokenizer(object):

    def __init__(self, langs, udpipe_models=None):
        self.models = {}
        if udpipe_models is not None:
            assert len(langs) == len(udpipe_models)
        for i, l in enumerate(langs):
            if udpipe_models:
                self.models[l] = Model(l, model_file=udpipe_models[i])
            else:
                self.models[l] = Model(l)

    def shutdown(self):
        pass

    def tokenize(self, text, lang):
        assert lang in self.models
        sentences = self.models[lang].tokenize(text, 'presegmented')
        for s in sentences:
            self.models[lang].tag(s)
            self.models[lang].parse(s)
        conllu = self.models[lang].write(sentences, "conllu")
        sentences = parse(conllu)
        outObj = OrderedDict([
            ('tokens', []),
            ('upos', []),
            ('head', []),
            ('deprel', [])
        ])

        # NOTE: num_tokens != num_words in a sentence, because tokens can be multi-word
        # we only use the first word's information for a multi-word token
        # IDEA CREDIT: https://github.com/ufal/udpipe/issues/123
        for idx, sentence in enumerate(sentences):
            tokens, upos, head, deprel = [], [], [], []
            word_to_token_map = {}
            for widx, word in enumerate(sentence):
                if isinstance(word['id'], tuple):
                    # multi-word token, e.g., word['id'] = (4, '-', 5)
                    assert len(word['id']) == 3
                    start, end = int(word['id'][0]), int(word['id'][2])
                    for word_id in list(range(start, end + 1)):
                        assert word_id not in word_to_token_map
                        word_to_token_map[word_id] = start
                else:
                    if word['misc'] is not None:
                        # single-word token
                        assert word['id'] not in word_to_token_map
                        word_to_token_map[word['id']] = word['id']
                    tokens.append(word['form'])
                    upos.append(word['upostag'])
                    deprel.append(word['deprel'])
                    assert isinstance(word['head'], int)
                    head.append(word['head'])

            assert len(tokens) == len(upos) == len(head) == len(deprel)
            outObj['tokens'].append(tokens)
            outObj['upos'].append(upos)
            outObj['head'].append(head)
            outObj['deprel'].append(deprel)

        return outObj

    def tokenize_pretokenized_sentence(self, tokens, lang):
        assert lang in self.models
        # My name is Wasi Ahmad
        # token_ranges = [[0, 2], [3, 7], [8, 10], [11, 15], [16, 21]]
        offset, token_ranges = 0, []
        for t in tokens:
            token_ranges.append([offset, offset + len(t)])
            offset += len(t) + 1

        sentences = self.models[lang].tokenize(' '.join(tokens), 'ranges;presegmented')
        for s in sentences:
            self.models[lang].tag(s)
            self.models[lang].parse(s)
        conllu = self.models[lang].write(sentences, "conllu")
        sentences = parse(conllu)
        assert len(sentences) == 1

        words, deptags, upos, heads, word_to_token = [], [], [], [], []
        _token_range = None
        for widx, word in enumerate(sentences[0]):
            word = sentences[0][widx]
            if word['misc'] is not None:
                _token_range = word['misc']['TokenRange'].split(':')
            start, end = int(_token_range[0]), int(_token_range[1])
            if isinstance(word['id'], tuple):
                # multi-word token, e.g., word['id'] = (4, '-', 5)
                pass
            else:
                words.append(word['form'])
                deptags.append(word['deprel'])
                upos.append(word['upostag'])
                assert isinstance(word['head'], int)
                heads.append(word['head'])
                match_indices = []
                # sometimes, during tokenization multiple tokens get merged
                # rect 230 550 300 620 Karl-Heinz Schnellinger
                # after tokenization
                # ['rect', '230 550 300 620', 'Karl-Heinz', 'Schnellinger']
                for j, o in enumerate(token_ranges):
                    if start >= o[0] and end <= o[1]:
                        match_indices.append(j)
                        break
                    elif start == o[0]:
                        match_indices.append(j)
                    elif end == o[1]:
                        match_indices.append(j)

                if len(match_indices) == 0:
                    return None
                word_to_token.append(match_indices[0])

        if len(words) != len(word_to_token):
            print(lang, tokens, words)
            assert False

        assert max(heads) <= len(heads)
        root, _ = head_to_tree(heads, words)
        # verifying if we can construct the tree from heads
        assert len(heads) == root.size()
        outObj = OrderedDict([
            ('tokens', words),
            ('deptag', deptags),
            ('upostag', upos),
            ('head', heads),
            ('word_to_token', word_to_token)
        ])

        return outObj


def xnli_pawsx_process(example):
    premise = TOK.tokenize(example['premise'], lang=LANGUAGES[0])
    hypothesis = TOK.tokenize(example['hypothesis'], lang=LANGUAGES[1])

    if len(premise['tokens']) > 0 and len(hypothesis['tokens']) > 0:
        return {
            'premise': {
                'text': example['premise'],
                'tokens': premise['tokens'][0],
                'upos': premise['upos'][0],
                'head': premise['head'][0],
                'deprel': premise['deprel'][0],
            },
            'hypothesis': {
                'text': example['hypothesis'],
                'tokens': hypothesis['tokens'][0],
                'upos': hypothesis['upos'][0],
                'head': hypothesis['head'][0],
                'deprel': hypothesis['deprel'][0],
            },
            'label': example['label']
        }
    else:
        return None


def xnli_pawsx_tokenization(infile, outfile, pre_lang, hyp_lang, workers=5):
    def load_dataset(path):
        """Load json file and store fields separately."""
        output = []
        with open(path) as f:
            for line in f:
                splits = line.strip().split('\t')
                if len(splits) != 3:
                    continue
                output.append({
                    'premise': splits[0],
                    'hypothesis': splits[1],
                    'label': splits[2]
                })
        return output

    global LANGUAGES
    LANGUAGES = [pre_lang, hyp_lang]
    pool = Pool(workers, initializer=init)

    processed_dataset = []
    dataset = load_dataset(infile)
    with tqdm(total=len(dataset), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(xnli_pawsx_process, dataset, 100)):
            pbar.update()
            if ex is not None:
                processed_dataset.append(ex)

    with open(outfile, 'w', encoding='utf-8') as fw:
        data_to_write = [json.dumps(ex, ensure_ascii=False) for ex in processed_dataset]
        fw.write('\n'.join(data_to_write))


def panx_process(example):
    sentence = TOK.tokenize_pretokenized_sentence(example['sentence'], LANGUAGES[0])
    if sentence is None:
        return None

    labels = []
    for i, tidx in enumerate(sentence['word_to_token']):
        labels.append(example['label'][tidx])

    assert len(sentence['tokens']) == len(labels)
    assert len(sentence['head']) == len(labels)

    return {
        'tokens': sentence['tokens'],
        'head': sentence['head'],
        'deptag': sentence['deptag'],
        'postag': sentence['upostag'],
        'label': labels
    }


def panx_tokenization(infile, outfile, pre_lang, workers=5, udpipe_model=None, separator='\t'):
    def load_dataset(path):
        """Load json file and store fields separately."""
        output = []
        with open(path, encoding='utf-8') as f:
            tokens, labels = [], []
            for line in f:
                splits = line.strip().split(separator)
                if len(splits) == 2:
                    tokens.append(splits[0])
                    labels.append(splits[1])
                else:
                    if tokens:
                        output.append({
                            'sentence': tokens,
                            'label': labels
                        })
                    tokens, labels = [], []

            if tokens:
                output.append({
                    'sentence': tokens,
                    'label': labels
                })

        return output

    processed_dataset = []
    dataset = load_dataset(infile)

    global LANGUAGES, UDPIPE_MODELS
    LANGUAGES = [pre_lang]
    UDPIPE_MODELS = [udpipe_model]
    pool = Pool(workers, initializer=init)

    desc_msg = '[{}] Processing'.format(pre_lang)
    with tqdm(total=len(dataset), desc=desc_msg) as pbar:
        for i, ex in enumerate(pool.imap(panx_process, dataset, 100)):
            pbar.update()
            if ex is not None:
                processed_dataset.append(ex)

    assert len(processed_dataset) <= len(dataset)
    if len(processed_dataset) < len(dataset):
        print('{} out of {} examples are discarded'.format(
            len(dataset) - len(processed_dataset), len(dataset)
        ))

    with open(outfile, 'w', encoding='utf-8') as fw:
        for ex in processed_dataset:
            assert len(ex['tokens']) == len(ex['label']) == len(ex['head'])
            fw.write(json.dumps(ex) + '\n')


def mtop_process(example):
    # {
    #     'tokens': words,
    #     'slot_labels': bio_tags,
    #     'intent_label': intent
    # }
    sentence = TOK.tokenize_pretokenized_sentence(example['tokens'], LANGUAGES[0])
    if sentence is None:
        return None

    labels = []
    for i, tidx in enumerate(sentence['word_to_token']):
        labels.append(example['slot_labels'][tidx])

    assert len(sentence['tokens']) == len(labels)
    assert len(sentence['head']) == len(labels)

    return {
        'tokens': sentence['tokens'],
        'deptag': sentence['deptag'],
        'postag': sentence['upostag'],
        'head': sentence['head'],
        'slot_labels': labels,
        'intent_label': example['intent_label']
    }


def mtop_tokenization(infile, outfile, pre_lang, workers=5):
    processed_dataset = []
    with open(infile) as f:
        dataset = [json.loads(line.strip()) for line in f]

    global LANGUAGES
    LANGUAGES = [pre_lang]
    pool = Pool(workers, initializer=init)

    desc_msg = '[{}] Processing'.format(pre_lang)
    with tqdm(total=len(dataset), desc=desc_msg) as pbar:
        for i, ex in enumerate(pool.imap(mtop_process, dataset, 100)):
            pbar.update()
            if ex is not None:
                processed_dataset.append(ex)

    assert len(processed_dataset) <= len(dataset)
    if len(processed_dataset) < len(dataset):
        print('{} out of {} examples are discarded'.format(
            len(dataset) - len(processed_dataset), len(dataset)
        ))

    with open(outfile, 'w', encoding='utf-8') as fw:
        for ex in processed_dataset:
            fw.write(json.dumps(ex) + '\n')


def matis_tokenization(infile, outfile, pre_lang, workers=5):
    processed_dataset = []
    dataset = []
    mismatch = 0
    with open(infile) as f:
        next(f)
        for line in f:
            split = line.strip().split('\t')
            tokens = split[1].split()
            slot_labels = split[2].split()
            if len(tokens) != len(slot_labels):
                if len(tokens) != len(slot_labels):
                    mismatch += 1
                    # print(split[0], tokens, slot_labels, len(tokens), len(slot_labels))
                    continue
            dataset.append({
                'tokens': tokens,
                'slot_labels': slot_labels,
                'intent_label': split[3]
            })

    print('{} examples are discarded due to mismatch in #tokens and #slot_labels'.format(mismatch))

    global LANGUAGES
    LANGUAGES = [pre_lang]
    pool = Pool(workers, initializer=init)

    desc_msg = '[{}] Processing'.format(pre_lang)
    with tqdm(total=len(dataset), desc=desc_msg) as pbar:
        for i, ex in enumerate(pool.imap(mtop_process, dataset, 100)):
            pbar.update()
            if ex is not None:
                processed_dataset.append(ex)

    assert len(processed_dataset) <= len(dataset)
    if len(processed_dataset) < len(dataset):
        print('{} out of {} examples are discarded'.format(
            len(dataset) - len(processed_dataset), len(dataset)
        ))

    with open(outfile, 'w', encoding='utf-8') as fw:
        for ex in processed_dataset:
            fw.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    languages = ['af', 'ar', 'bg', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
                 'he', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'mr', 'nl', 'pt',
                 'ru', 'ta', 'te', 'tr', 'ur', 'vi', 'zh']
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Path of the source data file")
    parser.add_argument('--output_file', type=str, required=True, help="Path of the processed data file")
    parser.add_argument('--pre_lang', type=str, help="Premise language", default='en', choices=languages)
    parser.add_argument('--hyp_lang', type=str, help="Hypothesis language", default='en', choices=languages)
    parser.add_argument('--task', type=str, default='pawsx', help="Task name",
                        choices=['pawsx', 'xnli', 'panx', 'mtop', 'matis', 'ner'])
    parser.add_argument('--tokenizer', type=str, default='udpipe', choices=['udpipe'],
                        help="How to perform tokenization")
    parser.add_argument('--udpipe_model', type=str, default=None,
                        help="Path of the UDPipe model")
    parser.add_argument('--workers', type=int, default=60)
    args = parser.parse_args()

    if args.tokenizer == 'udpipe':
        if args.task in ['pawsx', 'xnli']:
            xnli_pawsx_tokenization(
                args.input_file, args.output_file, args.pre_lang, args.hyp_lang, args.workers
            )
        elif args.task == 'panx':
            panx_tokenization(
                args.input_file, args.output_file, args.pre_lang, args.workers, args.udpipe_model
            )
        elif args.task == 'ner':
            panx_tokenization(
                args.input_file, args.output_file, args.pre_lang, args.workers,
                args.udpipe_model, ' '
            )
        elif args.task == 'mtop':
            mtop_tokenization(
                args.input_file, args.output_file, args.pre_lang, args.workers
            )
        elif args.task == 'matis':
            matis_tokenization(
                args.input_file, args.output_file, args.pre_lang, args.workers
            )
