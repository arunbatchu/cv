'''
Design. Take a given labeled dataset and split into train, valid, test folders.
Use provided ratios. For e.g 5:2:1  would mean move 5 records to train, 2 to valid, 1 to test. Repeat cycle until done.
'''
import os
import shutil

filelist = [f for f in os.listdir() if os.path.isfile(f)]
os.makedirs('valid')
os.makedirs('test')
os.makedirs('train')

valid_count=0
train_count=0
test_count=0
#Thresholds
train_threshold = 7
valid_threshold =2
test_threshold=1
for file in filelist:


    if train_count < train_threshold:
        shutil.move(file,'train')
        train_count += 1
    elif valid_count < valid_threshold:
        shutil.move(file,'valid')
        valid_count += 1
    elif test_count < test_threshold:
        test_count+=1
        shutil.move(file,'test')
    else:
        shutil.move(file, 'train')
        train_count =1
        valid_count=0
        test_count =0