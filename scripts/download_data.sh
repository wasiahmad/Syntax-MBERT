#!/bin/bash
# Copyright 2020 Google and DeepMind.
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

REPO=$PWD
DIR=$REPO/download
mkdir -p $DIR

# download XNLI dataset
function download_xnli {
    OUTPATH=$DIR/xnli-tmp
    if [[ ! -d $OUTPATH/XNLI-MT-1.0 ]]; then
        if [[ ! -f $OUTPATH/XNLI-MT-1.0.zip ]]; then
            wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
        fi
        unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [[ ! -d $OUTPATH/XNLI-1.0 ]]; then
        if [[ ! -f $OUTPATH/XNLI-1.0.zip ]]; then
            wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
        fi
        unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi
    python $REPO/utils_preprocess.py \
        --data_dir $OUTPATH \
        --output_dir $DIR/xnli \
        --task xnli;
    rm -rf $OUTPATH
    echo "Successfully downloaded data at $DIR/xnli" >> $DIR/download.log
}

# download PAWS-X dataset
function download_pawsx {
    cd $DIR
    wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz -q --show-progress
    tar xzf x-final.tar.gz -C $DIR
    python $REPO/utils_preprocess.py \
        --data_dir $DIR/x-final \
        --output_dir $DIR/pawsx \
        --task pawsx;
    rm -rf x-final x-final.tar.gz
    echo "Successfully downloaded data at $DIR/pawsx" >> $DIR/download.log
}

# download UD-POS dataset
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll
    mkdir -p $out_dir
    cd $base_dir
    curl -s --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
    tar -xzf $base_dir/ud-treebanks-v2.5.tgz

    langs=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh)
    for x in $base_dir/ud-treebanks-v2.5/*/*.conllu; do
        file="$(basename $x)"
        IFS='_' read -r -a array <<< "$file"
        lang=${array[0]}
        if [[ " ${langs[@]} " =~ " ${lang} " ]]; then
            lang_dir=$out_dir/$lang
            mkdir -p $lang_dir
            y=$lang_dir/${file/conllu/conll}
            if [[ ! -f "$y" ]]; then
                echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y \
                    --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
                python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y \
                    --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms;
            else
                echo "${y} exists"
            fi
        fi
    done

    python $REPO/utils_preprocess.py --data_dir $out_dir --output_dir $DIR/udpos --task udpos
    rm -rf $out_dir ud-treebanks-v2.tgz $DIR/udpos-tmp
    echo "Successfully downloaded data at $DIR/udpos" >> $DIR/download.log
}

function download_panx {
    echo "Download panx NER dataset"
    if [[ ! -f $DIR/AmazonPhotos.zip ]]; then
        echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive mannually and save it to $DIR/AmazonPhotos.zip"
        echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
    else
        base_dir=$DIR/panx_dataset
        unzip -qq -j $DIR/AmazonPhotos.zip -d $base_dir
        cd $base_dir
        langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu)
        for lg in ${langs[@]}; do
            tar xzf $base_dir/${lg}.tar.gz
            for f in dev test train; do
                mv $base_dir/$f $base_dir/${lg}-${f};
            done
        done
        cd ..
        python $REPO/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DIR/panx \
            --task panx;
        rm -rf $base_dir
        echo "Successfully downloaded data at $DIR/panx" >> $DIR/download.log
    fi
}

function download_mtop () {
    cd $DIR
    wget https://dl.fbaipublicfiles.com/mtop/mtop.zip -q --show-progress
    unzip mtop.zip -d .
    python $REPO/utils_preprocess.py \
        --data_dir $DIR/mtop \
        --output_dir $DIR/mtop \
        --task mtop;
    rm -rf mtop.zip
    echo "Successfully downloaded data at $DIR/mtop" >> $DIR/download.log
}


download_xnli
download_pawsx
download_udpos
download_panx
#download_mtop
