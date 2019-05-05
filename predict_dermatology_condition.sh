#!/usr/bin/env bash
python predict.py --imagepath $1  --top_k=3 --cat_to_name=dermatology_cat_to_name.json && eog $1
