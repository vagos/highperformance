#!/bin/env bash

ANTS=10000
GRID=1024
STEPS=1000
for GRID in 1024 2048 4096
do
    echo "Running ants with $ANTS ants, $GRID grid, $STEPS steps"
    ./ants-serial -a $ANTS -n $GRID -s $STEPS
    ./ants-fast   -a $ANTS -n $GRID -s $STEPS
done
