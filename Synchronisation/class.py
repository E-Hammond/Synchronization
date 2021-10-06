

class Synchronise():

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