import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# File path
cross_mag_file_path = 'Cross Spectrum Final proj/Data/Anechoic_500/500hz_cross_mag.txt'
cross_phase_file_path = 'Cross Spectrum Final proj/Data/Anechoic_500/500hz_cross_phase.txt'
fft_file_path = 'Cross Spectrum Final proj/Data/Anechoic_500/500hz_fft.txt'

# Read the data, skipping the first line (header) and setting column names
cross_mag_data = pd.read_csv(cross_mag_file_path, delim_whitespace=True, header=1, names=["Frequency", "Amplitude"])
cross_phase_data = pd.read_csv(cross_phase_file_path, delim_whitespace=True, header=1, names=["Frequency", "Phase"])
fft_data = pd.read_csv(fft_file_path, delim_whitespace=True, header=1, names=["Frequency1", "Microphone", "Frequency2", "Accelerometer"])

# Combine magnitude and phase data into a single DataFrame
cross_data = pd.merge(cross_mag_data, cross_phase_data, on="Frequency", how="inner") 

# Convert the columns to numeric values, coercing errors to NaN
cross_data["Frequency"] = pd.to_numeric(cross_data["Frequency"], errors='coerce')
cross_data["Amplitude"] = pd.to_numeric(cross_data["Amplitude"], errors='coerce')
cross_data["Phase"] = pd.to_numeric(cross_data["Phase"], errors='coerce')
fft_data["Frequency1"] = pd.to_numeric(fft_data["Frequency1"], errors='coerce')
fft_data["Microphone"] = pd.to_numeric(fft_data["Microphone"], errors='coerce')
fft_data["Frequency2"] = pd.to_numeric(fft_data["Frequency2"], errors='coerce')
fft_data["Accelerometer"] = pd.to_numeric(fft_data["Accelerometer"], errors='coerce')

# Drop rows with NaN values
cross_data = cross_data.dropna()
fft_data = fft_data.dropna()

# Filter the data to only include frequencies up to 5000 Hz
cross_data = cross_data[cross_data["Frequency"] <= 5000]
fft_data = fft_data[fft_data["Frequency1"] <= 5000]

# Plot cross spectrum magnitude and phase
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(cross_data["Frequency"], cross_data["Amplitude"], color='blue', linewidth=0.5)
plt.title('Cross Spectrum Magnitude')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(cross_data["Frequency"], cross_data["Phase"], color='orange', linewidth=0.5)
plt.title('Cross Spectrum Phase')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot FFT data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(fft_data["Frequency1"], fft_data["Microphone"], label='FFT Microphone', color='green', linewidth=0.5)
plt.title('FFT Microphone')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(fft_data["Frequency1"], fft_data["Accelerometer"], label='FFT Accelerometer', color='red', linewidth=0.5)
plt.title('FFT Accelerometer')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




