#!/bin/bash
#DSUB -n zhy
#DSUB -N 1
#DSUB -A root.qrn8y2ug
#DSUB -R "cpu=32;gpu=1;mem=240000"
#DSUB -oo %J.out
#DSUB -eo %J.err

python cli.py
