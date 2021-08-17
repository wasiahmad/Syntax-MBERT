# Syntax-augmented Multilingual BERT
Official code release of our ACL 2021 work, [Syntax-augmented Multilingual BERT for Cross-lingual Transfer](https://aclanthology.org/2021.acl-long.350).

**[Notes]**

- This repository provides implementations for three NLP applications.
    - Text classification, named entity recognition, and task-oriented semantic parsing. 
- We will release the question answering (QA) model implementation soon. 


### Setup
We setup a conda environment in order to run experiments. We assume [anaconda](https://www.anaconda.com/) 
and Python 3.6 is installed. The additional requirements (as noted in requirements.txt can be installed by running 
the following script:

```bash
bash install_tools.sh
```


### Data Preparation

The next step is to download the data. To this end, first create a `download` folder with `mkdir -p download` in the root 
of this project. You then need to manually download `panx_dataset` (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN)
(note that it will download as `AmazonPhotos.zip`) to the download directory. Finally, run the following command to 
download the remaining datasets:

```bash
bash scripts/download_data.sh
```

To get the POS-tags and dependency parse of input sentences, we use UDPipe. Go to the 
[udpipe](https://github.com/wasiahmad/Syntax-MBERT/tree/main/udpipe) directory and run the task-specific scripts -
`[xnli.sh|pawsx.sh|panx.sh|mtop.sh]`.


### Training and Evaluation

The evaluation results (on the test set) are saved in `${SAVE_DIR}` directory (check the bash scripts).

#### Text Classification

```bash
cd scripts
bash xlt_classify.sh GPU TASK USE_SYNTAX SEED
```

For **cross-lingual** text classification, do the following.

```bash
# for XNLI
bash xlt_classify.sh 0 xnli false 1111

# for PAWS-X
bash xlt_classify.sh 0 pawsx false 1111
```

- For syntax-agumented MBERT experiments, set `USE_SYNTAX=true`.
- For **generalized cross-lingual** text classification evaluation, use the 
[gxlt_classify.sh](https://github.com/wasiahmad/Syntax-MBERT/blob/main/scripts/gxlt_classify.sh) script.


#### Named Entity Recognition

```bash
cd scripts
bash panx.sh GPU USE_SYNTAX SEED
```

- For syntax-agumented MBERT experiments, set `USE_SYNTAX=true`.
- For the CoNLL NER datasets, same set of scripts can be used (with revision). 


#### Task-oriented Semantic Parsing

```bash
cd scripts
bash mtop.sh GPU USE_SYNTAX SEED
```

- For syntax-agumented MBERT experiments, set `USE_SYNTAX=true`.
- Since, mATIS++ dataset is not publicly available, we do not release the scripts.


### Acknowledgement
We acknowledge the efforts of the authors of the following repositories.

- https://github.com/google-research/xtreme
- https://github.com/huggingface/transformers


### Citation

```
@inproceedings{ahmad-etal-2021-syntax,
    title = "Syntax-augmented Multilingual {BERT} for Cross-lingual Transfer",
    author = "Ahmad, Wasi  and
      Li, Haoran  and
      Chang, Kai-Wei  and
      Mehdad, Yashar",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.350",
    pages = "4538--4554",
}
```
