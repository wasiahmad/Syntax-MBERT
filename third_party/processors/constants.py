NUM_SPECIAL_TOKENS = 4
BERT_SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
XLMR_SPECIAL_TOKENS = ['<pad>', '<s>', '</s>', '<unk>']

POS_SYMBOLS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X"
]

DEPTAG_SYMBOLS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj",
    "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj",
    "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum",
    "root", "vocative", "xcomp"
]


def UPOS_MAP(tokenizer):
    sp_tokens = XLMR_SPECIAL_TOKENS if 'roberta' in tokenizer else BERT_SPECIAL_TOKENS
    symbols = sp_tokens[:1] + POS_SYMBOLS + sp_tokens[1:]
    return {s: idx for idx, s in enumerate(symbols)}


def DEP_TAG_MAP(tokenizer):
    sp_tokens = XLMR_SPECIAL_TOKENS if 'roberta' in tokenizer else BERT_SPECIAL_TOKENS
    symbols = sp_tokens[:1] + DEPTAG_SYMBOLS + sp_tokens[1:] + ['<self>']
    return {s: idx for idx, s in enumerate(symbols)}


def upos_to_id(param, tokenizer):
    use_map = UPOS_MAP(tokenizer)
    if isinstance(param, str):
        return use_map[param]
    elif isinstance(param, list):
        return [use_map[i] for i in param]
    else:
        raise ValueError()


def deptag_to_id(param, tokenizer):
    use_map = DEP_TAG_MAP(tokenizer)
    unk_sym = '<unk>' if 'roberta' in tokenizer else '[UNK]'
    if isinstance(param, str):
        return use_map.get(param, use_map[unk_sym])
    elif isinstance(param, list):
        return [use_map.get(i, use_map[unk_sym]) for i in param]
    else:
        raise ValueError()
