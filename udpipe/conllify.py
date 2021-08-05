import ufal.udpipe

# UDPipe supports all MLQA and PAWS-X languages but only 13/15 languages for XNLI
LANG_MAP = {
    'ar': 'models/arabic-padt-ud-2.5-191206.udpipe',  # MLQA, XNLI
    'bg': 'models/bulgarian-btb-ud-2.5-191206.udpipe',  # XNLI
    'de': 'models/german-gsd-ud-2.5-191206.udpipe',  # MLQA, PAWS-X, XNLI
    'el': 'models/greek-gdt-ud-2.5-191206.udpipe',  # XNLI
    'en': 'models/english-ewt-ud-2.5-191206.udpipe',  # MLQA, XNLI
    'es': 'models/spanish-gsd-ud-2.5-191206.udpipe',  # MLQA, PAWS-X, XNLI
    'fr': 'models/french-gsd-ud-2.5-191206.udpipe',  # PAWS-X, XNLI
    'hi': 'models/hindi-hdtb-ud-2.5-191206.udpipe',  # MLQA, XNLI
    'ja': 'models/japanese-gsd-ud-2.5-191206.udpipe',  # PAWS-X, 
    'ko': 'models/korean-gsd-ud-2.5-191206.udpipe',  # PAWS-X, 
    'ru': 'models/russian-gsd-ud-2.5-191206.udpipe',  # XNLI
    'tr': 'models/turkish-imst-ud-2.5-191206.udpipe',  # XNLI
    'ur': 'models/urdu-udtb-ud-2.5-191206.udpipe',  # XNLI
    'vi': 'models/vietnamese-vtb-ud-2.5-191206.udpipe',  # MLQA, XNLI
    'zh': 'models/chinese-gsd-ud-2.5-191206.udpipe',  # MLQA, PAWS-X, XNLI
    'pt': 'models/portuguese-gsd-ud-2.5-191206.udpipe',
}


class Model:
    def __init__(self, lang, model_file=None):
        """Load given model."""
        self.lang = lang
        if model_file:
            self.model = ufal.udpipe.Model.load(model_file)
            if not self.model:
                raise Exception("Cannot load UDPipe model from file '%s'" % model_file)
        else:
            self.model = ufal.udpipe.Model.load(LANG_MAP[lang])
            if not self.model:
                raise Exception("Cannot load UDPipe model from file '%s'" % LANG_MAP[lang])

    def tokenize(self, text, *args):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(*args)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output
