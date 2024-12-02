#!/bin/bash

#subsets='affordance ownership task recommendation proxemics emotion'
#subsets='emotion'
subsets='task recommendation proxemics emotion'

for subset in $subsets
do
    python main.py scores $subset qwen2_7b
    #python main.py scores $subset llama31_8b
    #python main.py scores $subset dummy
    #python main.py sequences $subset mistral7b
done

