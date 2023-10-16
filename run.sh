#!/bin/bash

subsets='affordance ownership task recommendation proxemics emotion'

for subset in $subsets
do
    python main.py scores $subset falcon
    python main.py sequences $subset falcon
done

#subsets='recommendation emotion proxemics affordance ownership'
#
#for subset in $subsets
#do
#    python main.py scores $subset open_llama
#    python main.py sequences $subset open_llama
#done
