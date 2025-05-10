import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

'''
Read in the data
'''
#File path
time_data_file_path = 'Cross Spectrum Final proj/Data/Anechoic_269hz_speaker_ground/time_domain.txt'
fft_mag_file_path = 'Cross Spectrum Final proj/Data/Anechoic_269hz_speaker_ground/fft_mag.txt'
fft_phase_file_path = 'Cross Spectrum Final proj/Data/Anechoic_269hz_speaker_ground/fft_phase.txt'

# Read the fft data, skipping the first line (header) and setting column names
fft_mag_data = pd.read_csv(fft_mag_file_path, delim_whitespace=True, skiprows=1, names=["Frequency1", "Microphone", "Frequency2", "Accelerometer"])
fft_phase_data = pd.read_csv(fft_phase_file_path, delim_whitespace=True, skiprows=1, names=["Frequency1", "Microphone", "Frequency2", "Accelerometer"])

# Read the time data skip the first line (header)
# Also delete the third column because it is redundant
time_data = pd.read_csv(time_data_file_path, delim_whitespace=True, skiprows=1, names=["Time", "Microphone", "Time2", "Accelerometer"])
time_data = time_data.drop(columns=["Time2"])



'''
Further processing of the data
'''
# Get the cross spectrum
cross_frequencies, cross_spectrum = signal.csd(time_data["Microphone"], time_data["Accelerometer"], fs=51200, nperseg=16384)
cross_magnitude = np.abs(cross_spectrum)
cross_phase = np.angle(cross_spectrum, deg=True)  # Convert to degrees
cross_phase = np.mod(cross_phase, 360)  # Ensure phase is between 0 and 360 degrees
cross_phase = cross_phase - 180  # Shift phase by -180 degrees

# Get the coherence function
coherence_frequencies, coherence = signal.coherence(time_data["Microphone"], time_data["Accelerometer"], fs=51200, nperseg=16384)



'''
Plotting the data
'''
# Plot FFT data
# Cut data at 5000 Hz
fft_mag_data = fft_mag_data[fft_mag_data["Frequency1"] <= 5000]
fft_phase_data = fft_phase_data[fft_phase_data["Frequency1"] <= 5000]
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(fft_mag_data["Frequency1"], fft_mag_data["Microphone"], label='FFT Microphone', color='green', linewidth=0.5)
plt.title('FFT Microphone')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(fft_mag_data["Frequency1"], fft_mag_data["Accelerometer"], label='FFT Accelerometer', color='red', linewidth=0.5)
plt.title('FFT Accelerometer')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/anechoic_269hz_speaker_ground/fft.png')


# Plot cross spectrum magnitude and phase
# Only plot up to 5000 Hz
cross_frequencies = cross_frequencies[cross_frequencies <= 5000]
cross_magnitude = cross_magnitude[:len(cross_frequencies)]
cross_phase = cross_phase[:len(cross_frequencies)]
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(cross_frequencies, cross_magnitude, color='blue', linewidth=0.5)
plt.title('Cross Spectrum Magnitude')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(cross_frequencies, cross_phase, color='red', linewidth=0.5)
plt.title('Cross Spectrum Phase')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/anechoic_269hz_speaker_ground/cross.png')

# Plot coherence
#Only plot up to 5000 Hz
coherence_frequencies = coherence_frequencies[coherence_frequencies <= 5000]
coherence = coherence[:len(coherence_frequencies)]
plt.figure(figsize=(10, 6))
plt.plot(coherence_frequencies, coherence, color='green', linewidth=0.5)
plt.title('Coherence')
plt.ylabel('Coherence')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/anechoic_269hz_speaker_ground/coherence.png')


