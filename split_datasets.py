'''
Design. Take a given labeled dataset and split into train, valid, test folders.
Use provided ratios. For e.g 5:2:1  would mean move 5 records to train, 2 to valid, 1 to test. Repeat cycle until done.
'''
import os
import shutil
os.makedirs('valid')
os.makedirs('test')
os.makedirs('train')