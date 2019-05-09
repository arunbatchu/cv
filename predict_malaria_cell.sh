#!/usr/bin/env bash
python predict.py --imagepath $1  --top_k=2 --cat_to_name=malaria_cell_cat_to_name.json && eog $1
