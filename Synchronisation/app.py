import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import biosignalsnotebooks as bsnb
from sklearn.preprocessing import MinMaxScaler
import os

from sync import *


#declare all args
pred_csv_path = r'.\\Datasets\\mmm4.csv'
actual_csv_url = r'.\\Datasets\\pide4_2021-07-23_14_23_02_my_iOS_device.csv'


#read when speed network is done 
predict_csv= pd.read_csv(pred_csv_path)
actual_csv = pd.read_csv(actual_csv_url)




#sync data 
Sync_Signals(actual_csv["locationSpeed(m/s)"], predict_csv["speed"])

