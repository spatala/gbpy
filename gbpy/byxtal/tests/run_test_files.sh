#!/bin/zsh

for py_file in $(ls *.py)
do
    python $py_file
done