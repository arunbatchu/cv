#!/usr/bin/env bash
python predict.py --imagepath $1  --top_k=5 --cat_to_name=cat_to_name.json && eog $1
