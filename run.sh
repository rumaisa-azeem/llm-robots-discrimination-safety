#!/bin/bash

subsets='affordance ownership task recommendation proxemics emotion'
#subsets='emotion'

for subset in $subsets
do
    python main.py scores $subset vicuna13b
    python main.py sequences $subset vicuna13b
done

