#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    prepare)
        echo "download Naver Movie Corpus..."
        wget https://github.com/e9t/nsmc/raw/master/ratings.txt
        mkdir data
        mv ratings.txt data
        ;;
    train)
        echo "train Convolutional Neural Networks..."
        python train.py preprocess
        python train.py train
        ;;
    web-demo)
        echo "web demo"
        python app.py
        ;;
esac