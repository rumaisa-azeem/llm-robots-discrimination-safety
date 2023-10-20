#!/bin/bash

subsets='affordance ownership task recommendation proxemics emotion'

for subset in $subsets
do
    python main.py scores $subset mistral
    python main.py sequences $subset mistral
done

