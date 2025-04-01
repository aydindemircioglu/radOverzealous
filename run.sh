#!/bin/bash

python3 ./experiment.py --cohort=radMLBench
python3 ./experiment.py --cohort=UCI

python3 ./evaluate.py
