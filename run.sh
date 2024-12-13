#!/bin/bash

#subsets='affordance ownership task recommendation proxemics emotion'
#subsets='emotion'
subsets='task recommendation proxemics emotion'

for subset in $subsets
do
    python main.py scores $subset mistral7b
    python main.py scores $subset llama31_8b
done

