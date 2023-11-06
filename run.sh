#!/bin/bash

subsets='affordance ownership task recommendation proxemics emotion'

for subset in $subsets
do
    python main.py scores $subset mistral7b
    python main.py sequences $subset mistral7b
done

