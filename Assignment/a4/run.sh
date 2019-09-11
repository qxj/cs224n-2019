#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
    mkdir outputs
    touch outputs/test_outputs.txt
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
elif [ "$1" = "train_zh" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./ch_en_data/train.ch --train-tgt=./ch_en_data/train.en --dev-src=./ch_en_data/dev.ch --dev-tgt=./ch_en_data/dev.en --vocab=vocab_zh.json --save-to=model_zh.bin --cuda
elif [ "$1" = "test_zh" ]; then
    mkdir outputs_zh
    touch outputs/test_outputs.txt
        CUDA_VISIBLE_DEVICES=0 python run.py decode model_zh.bin ./ch_en_data/test.ch ./ch_en_data/test.en outputs_zh/test_outputs.txt --cuda
elif [ "$1" = "vocab_zh" ]; then
	python vocab.py --train-src=./ch_en_data/train.ch --train-tgt=./ch_en_data/train.en vocab_zh.json
else
	echo "Invalid Option Selected"
fi
