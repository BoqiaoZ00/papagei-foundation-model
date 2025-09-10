import vitaldb
import numpy as np
import matplotlib.pyplot as plt

# Load a specific case (example case id)
caseid = 100  # You would use a real case id from the dataset
tracks = ["SNUADC/PLETH"] # Common track names for PPG

# Load the waveforms at 100 Hz
data = vitaldb.load_case(caseid, tracks, 1/100)
print(data.shape)

# Separate the arrays
time = np.arange(0, len(data)) / 100  # create time axis in seconds
ppg_signal = data[:, 0]

# Plot a 10-second segment
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(time[1000:2000], ppg_signal[1000:2000], 'b')
plt.title('ECG Signal')


plt.tight_layout()
plt.show()