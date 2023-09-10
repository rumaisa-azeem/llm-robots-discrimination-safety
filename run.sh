#!/bin/bash

subsets='task recommendation emotion proxemics affordance ownership'

for subset in $subsets
do
    python main.py scores $subset falcon
    python main.py sequences $subset falcon
done

for subset in $subsets
do
    python main.py scores $subset open_llama
    python main.py sequences $subset open_llama
done