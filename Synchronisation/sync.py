import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import biosignalsnotebooks as bsnb
from sklearn.preprocessing import MinMaxScaler


### NB You should be reading from cloud path

### Reading CSV Files locally
# file_path = r'.\Datasets'
# # csv_1 = pd.read_csv(file_path + '\ground_truth.csv')
# predict_csv = pd.read_csv(file_path + '\mmm4.csv')
# actual_csv = pd.read_csv(file_path + '\pide4_2021-07-23_14_23_02_my_iOS_device.csv')



## Define a function to resample 30Hz speed to 10Hz
# def resample_signal(signal):
#     signal['Date'] = pd.to_datetime(signal['loggingTime(txt)'])
#     signal = signal.set_index(['Date'])
#     signal = signal.resample('0.1S').median()
#     signal = signal.reset_index()
    
#     return signal

# def normalise_signals(signal_1,signal_2):
    
#     scaler = MinMaxScaler()
#     signal_1 = np.array(ride2['locationSpeed(m/s)'])
#     signal_2 = np.array(motor['Speed'])
#     signal_1 = scaler.fit_transform(signal_1.reshape(-1,1))
#     signal_2 = scaler.fit_transform(signal_2.reshape(-1,1))

#     signal_1 = pd.DataFrame(signal_1)
#     signal_2 = pd.DataFrame(signal_2)
#     signal_1.columns = ['Speed']
#     signal_2.columns = ['Speed']


def interpolate_signal(data):
    data['time'] = data['loggingTime(txt)'].apply(lambda x: datetime.strptime(" ".join(x.split(" ")[0:-1]), '%Y-%m-%d %H:%M:%S.%f'))
    data=data.loc[:, ["time", 'locationSpeed(m/s)']]
    data = data.set_index("time")
    data = data[~data.index.duplicated(keep='first')].reset_index()
    data['unix_time'] = data['time'].apply(lambda x: x.timestamp())
    diff=data['time'].max()-data['time'].min()
    fps = 10
    new_frames = int(diff.total_seconds()*fps)
    interpolation_time = np.linspace(np.min(data['unix_time']),np.max(data['unix_time']), new_frames)

    interp = interp1d(data['unix_time'].values, data['locationSpeed(m/s)'].values, kind='cubic')
    interpolated_values=interp(interpolation_time)
    return interpolated_values


### Defining a function that takes two signals and synchronises them
def Sync_Signals(signal_1,signal_2):

    pred_csv_path = r'.\\Datasets\\mmm4.csv'
    predict_csv = predict_csv= pd.read_csv(pred_csv_path)

    ## When the length of Signal 1 is greater than Signal 2 
    if len(signal_1) > len(signal_2):
        dephase, signal_2, signal_1 = bsnb.synchronise_signals(signal_2,signal_1)

        ## When the length signal 1 is different from signal after sync
        if len(signal_1) != len(signal_2):
            lenght_diff = abs(len(signal_1) - len(signal_2))
            if len(signal_1) > len(signal_2):
                signal_1 = signal_1[:-lenght_diff]
            else:
                signal_2 = signal_2[:-lenght_diff]

            ## We reset index and use old index as columns
            signal_1 = pd.DataFrame(signal_1).reset_index()
            signal_1.columns = ['Actual_index','Actual']
            signal_2 = pd.DataFrame(signal_2).reset_index()
            signal_2.columns = ['Predicted_index','Predicted']
            data = pd.concat([signal_1,signal_2], axis=1)
            ### Retrieving the corresponding images from predict_csv
            predicted_index = [i for i in data['Predicted_index']]
            predict_images = predict_csv.iloc[predicted_index]
            data['images'] = predict_images['image']
            data.to_csv("Synced_Speed.csv", index=False)
            
            return data

        ## When the length signal 1 and signal 2 are the same    
        else:
            print('return the two signals which are equal')
            
            ## We reset index and use old index as columns
            signal_1 = pd.DataFrame(signal_1).reset_index()
            signal_1.columns = ['Actual_index','Actual']
            signal_2 = pd.DataFrame(signal_2).reset_index()
            signal_2.columns = ['Predicted_index','Predicted']
            data = pd.concat([signal_1,signal_2], axis=1)
            ### Retrieving the corresponding images from predict_csv
            predicted_index = [i for i in data['Predicted_index']]
            predict_images = predict_csv.iloc[predicted_index]
            data['images'] = predict_images['image']
            data.to_csv("Synced_Speed.csv", index=False)
            
            return data

    ## When the length of Signal 1 is smaller than Signal 2
    elif len(signal_1) < len(signal_2):
        
        dephase, signal_1, signal_2 = bsnb.synchronise_signals(signal_1,signal_2)

        ## When the length of signal 1 and signal 2 are different
        if (len(signal_1) != len(signal_2)):
            lenght_diff = abs(len(signal_1) - len(signal_2))
            if len(signal_1) > len(signal_2):
                signal_1 = signal_1[:-lenght_diff]
            if len(signal_1) < len(signal_2):
                signal_2 = signal_2[:-lenght_diff]
            
            ## We reset index and use old index as columns
            signal_1 = pd.DataFrame(signal_1).reset_index()
            signal_1.columns = ['Actual_index','Actual']
            signal_2 = pd.DataFrame(signal_2).reset_index()
            signal_2.columns = ['Predicted_index','Predicted']
            data = pd.concat([signal_1,signal_2], axis=1)
            ### Retrieving the corresponding images from predict_csv
            predicted_index = [i for i in data['Predicted_index']]
            predict_images = predict_csv.iloc[predicted_index]
            data['images'] = predict_images['image']
            data.to_csv("Synced_Speed.csv", index=False)
            
            return data

        ## When the length of signal 1 and signal 2 are the same     
        else:
            print('return the two signals which are equal')
            
            ## We reset index and use old index as columns
            signal_1 = pd.DataFrame(signal_1).reset_index()
            signal_1.columns = ['Actual_index','Actual']
            signal_2 = pd.DataFrame(signal_2).reset_index()
            signal_2.columns = ['Predicted_index','Predicted']
            data = pd.concat([signal_1,signal_2], axis=1)
            ### Retrieving the corresponding images from predict_csv
            predicted_index = [i for i in data['Predicted_index']]
            predict_images = predict_csv.iloc[predicted_index]
            data['images'] = predict_images['image']
            data.to_csv("Synced_Speed.csv", index=False)
            
            return data



# Sync_Signals(actual_csv['locationSpeed(m/s)'],predict_csv['speed'])