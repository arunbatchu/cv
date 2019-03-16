
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import csv

# Input (temp, rainfall, humidity)
from torch.utils.data import TensorDataset, DataLoader

inputs = np.array(
    [[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58],
     [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]],
    dtype='float32')
# Targets (apples, oranges) yield in tons
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119],
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')


inputs_df = pd.DataFrame(data=inputs, columns=['temperature','rainfall','humidity'],dtype='float32')
targets_df = pd.DataFrame(data=targets, columns=['apples','oranges'],dtype='float32')
inputs_targets_df = pd.concat([inputs_df,targets_df],axis=1)

inputs_targets_df.to_csv('weather_inputs.csv', sep=',' ,index=None)
#####################
# Read the csv file back
weather_data_df = pd.read_csv('weather_inputs.csv',sep=',')
print(weather_data_df.head(10))
targets_df = weather_data_df[['apples','oranges']]
print(targets_df)
print(torch.tensor(targets_df.values))