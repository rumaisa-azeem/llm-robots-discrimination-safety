#!/bin/bash

subsets='task recommendation emotion proxemics affordance ownership'

for subset in $subsets
do
    python main.py scores $subset falcon
done

for subset in $subsets
do
    python main.py scores $subset open_llama
done