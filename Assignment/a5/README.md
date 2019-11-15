# Assignment 5: Sub-word modeling & Convolutional Neural Network

[TOC]

#1

## a

$e_{word}=256$，$e_{char}=50$

Obviously, words are much more than characters.

## b

### char-based

1) 公式3 $ x_{emb} = CharEmbedding(x_{padded})$，有一组参数 => $V_{char}\times e_{char}$

2) 公式5 $x_{conv}=Conv1D(x_{reshape})$, 有两组参数：$W\in R^{f\times e_{char} \times k}$ 和 $b\in R^f$ => $e_{word}\times e_{char} \times k + e_{word}$

3) 公式8和9，有四组参数：$W_{proj}, b_{proj}, W_{gate}, b_{gate}$ => $2\times (e_{word}\times e_{word} + e_{word})$

### word-based

就一个embedding lookup，所以参数量是 $V_{word}\times e_{word}$。

如果，$k=5$，$V_{word}\approx 50000$，$V_{char}=96$，则

$N_{char}=(96\times 50)+(256\times 50\times 5+256)+2\times(256\times 256+256)=200640$

$N_{word}=50000\times 256=12800000$

## c

ConvNet多个filter可能捕捉到不同的local特征

## d

MaxPooling能捕捉到最强的pattern，但也会丢失其他较弱的信号，这在图像中无伤大雅，但在NLP里上下文更重要，可能会有问题。

# 2

train

```
epoch 29, iter 196310, avg. loss 84.85, avg. ppl 118.85 cum. examples 9920, speed 1500.96 words/sec, time elapsed 76321.95 sec
epoch 29, iter 196320, avg. loss 81.88, avg. ppl 115.76 cum. examples 10240, speed 1486.56 words/sec, time elapsed 76325.66 sec
epoch 29, iter 196330, avg. loss 89.18, avg. ppl 137.83 cum. examples 10537, speed 1430.67 words/sec, time elapsed 76329.42 sec
reached maximum number of epochs!
```

test

```
load test source sentences from [./en_es_data/test.es]
load test target sentences from [./en_es_data/test.en]
load model from model.bin
Decoding: 100%|█████████████████████████████| 8064/8064 [16:37<00:00,  8.08it/s]
Corpus BLEU: 24.588190642455157
```

