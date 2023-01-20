#!/bin/env bash

image()
{
    ROWS=$1
    COLS=$2
    IN_FILENAME=$3
    OUT_FILENAME=$4
    PCP=$5 # Principal Component Percentage

    K=$(($PCP * $COLS / 100))
    
    echo Image: $IN_FILENAME NPC: $K

    OUT_FILENAME="$OUT_FILENAME"_"$PCP"

    ./pca_omp -m $ROWS -n $COLS -npc $K -if $IN_FILENAME -of $OUT_FILENAME.txt
    octave ../pca_matlab/show_image.m $OUT_FILENAME.txt $OUT_FILENAME.jpg
    rm $OUT_FILENAME.txt
}

NPCs=(1 25 75 100)

for PC in ${NPCs[@]}; do

    # image 469 700 ../pca_data/elvis_new_ij.bin.gz out/elvis $PC
    image 4096 4096 ../pca_data/cyclone.bin.gz out/cyclone $PC
    # image 9500 9500 ../pca_data/earth.bin.gz out/earth $PC

done
