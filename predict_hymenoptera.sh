#!/usr/bin/env bash
python predict.py --imagepath $1  --top_k=2 --cat_to_name=ants_bees_cat_to_name.json && eog $1
