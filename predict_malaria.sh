#!/usr/bin/env bash
python predict.py --imagepath $1  --top_k=2 --cat_to_name=malaria_cell_cat_to_name.json && eog $1


#malaria/test/infected/C68P29N_ThinF_IMG_20150819_134504_cell_125.png
#malaria/test/infected/C68P29N_ThinF_IMG_20150819_134504_cell_124.png
#malaria/test/infected/C97P58ThinF_IMG_20150917_151903_cell_26.png
#malaria/test/infected/C98P59ThinF_IMG_20150917_153030_cell_158.png