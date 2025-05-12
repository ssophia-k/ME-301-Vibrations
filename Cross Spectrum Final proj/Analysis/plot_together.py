import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

'''
Read in the data
'''
#File path
speaker_ground_file_path = 'Cross Spectrum Final proj/Data/Anechoic_269hz_speaker_ground/time_domain.txt'
speaker_lifted_file_path = 'Cross Spectrum Final proj/Data/Anechoic_269hz_speaker_lifted/time_domain.txt'
anechoic_file_path = 'Cross Spectrum Final proj/Data/Anechoic/time_domain.txt'
brooks_file_path = 'Cross Spectrum Final proj/Data/Brooks/time_domain.txt'

# Read the time data skip the first line (header)
# Also delete the third column because it is redundant
speaker_ground_data = pd.read_csv(speaker_ground_file_path, delim_whitespace=True, skiprows=1, names=["Time", "Microphone", "Time2", "Accelerometer"])
speaker_ground_data = speaker_ground_data.drop(columns=["Time2"])
speaker_lifted_data = pd.read_csv(speaker_lifted_file_path, delim_whitespace=True, skiprows=1, names=["Time", "Microphone", "Time2", "Accelerometer"])
speaker_lifted_data = speaker_lifted_data.drop(columns=["Time2"])
anechoic_data = pd.read_csv(anechoic_file_path, delim_whitespace=True, skiprows=1, names=["Time", "Microphone", "Time2", "Accelerometer"])
anechoic_data = anechoic_data.drop(columns=["Time2"])
brooks_data = pd.read_csv(brooks_file_path, delim_whitespace=True, skiprows=1, names=["Time", "Microphone", "Time2", "Accelerometer"])
brooks_data = brooks_data.drop(columns=["Time2"])



'''
Further processing of the data
'''
# Get the cross spectrum
speaker_ground_frequencies, speaker_ground_spectrum = signal.csd(speaker_ground_data["Microphone"], speaker_ground_data["Accelerometer"], fs=51200, nperseg=16384)
speaker_ground_magnitude = np.abs(speaker_ground_spectrum)
speaker_lifted_frequencies, speaker_lifted_spectrum = signal.csd(speaker_lifted_data["Microphone"], speaker_lifted_data["Accelerometer"], fs=51200, nperseg=16384)
speaker_lifted_magnitude = np.abs(speaker_lifted_spectrum)
anechoic_frequencies, anechoic_spectrum = signal.csd(anechoic_data["Microphone"], anechoic_data["Accelerometer"], fs=51200, nperseg=16384)
anechoic_magnitude = np.abs(anechoic_spectrum)
brooks_frequencies, brooks_spectrum = signal.csd(brooks_data["Microphone"], brooks_data["Accelerometer"], fs=51200, nperseg=16384)
brooks_magnitude = np.abs(brooks_spectrum)

# Get cross spectrum phases
speaker_ground_phase = np.angle(speaker_ground_spectrum, deg=True)  # Convert to degrees
speaker_ground_phase = np.mod(speaker_ground_phase, 360)  # Ensure phase is between 0 and 360 degrees
speaker_ground_phase = speaker_ground_phase - 180  # Shift phase by -180 degrees
speaker_lifted_phase = np.angle(speaker_lifted_spectrum, deg=True)  # Convert to degrees
speaker_lifted_phase = np.mod(speaker_lifted_phase, 360)  # Ensure phase is between 0 and 360 degrees
speaker_lifted_phase = speaker_lifted_phase - 180  # Shift phase by -180 degrees
anechoic_phase = np.angle(anechoic_spectrum, deg=True)  # Convert to degrees
anechoic_phase = np.mod(anechoic_phase, 360)  # Ensure phase is between 0 and 360 degrees
anechoic_phase = anechoic_phase - 180  # Shift phase by -180 degrees
brooks_phase = np.angle(brooks_spectrum, deg=True)  # Convert to degrees
brooks_phase = np.mod(brooks_phase, 360)  # Ensure phase is between 0 and 360 degrees
brooks_phase = brooks_phase - 180  # Shift phase by -180 degrees


# Get the coherence function
speaker_ground_coherence_frequencies, speaker_ground_coherence = signal.coherence(speaker_ground_data["Microphone"], speaker_ground_data["Accelerometer"], fs=51200, nperseg=16384)
speaker_lifted_coherence_frequencies, speaker_lifted_coherence = signal.coherence(speaker_lifted_data["Microphone"], speaker_lifted_data["Accelerometer"], fs=51200, nperseg=16384)
anechoic_coherence_frequencies, anechoic_coherence = signal.coherence(anechoic_data["Microphone"], anechoic_data["Accelerometer"], fs=51200, nperseg=16384)
brooks_coherence_frequencies, brooks_coherence = signal.coherence(brooks_data["Microphone"], brooks_data["Accelerometer"], fs=51200, nperseg=16384)

#Plot all cross spectra on top of each other
# Only plot up to 5000 Hz
speaker_ground_frequencies = speaker_ground_frequencies[speaker_ground_frequencies <= 5000]
speaker_ground_magnitude = speaker_ground_magnitude[:len(speaker_ground_frequencies)]
speaker_lifted_frequencies = speaker_lifted_frequencies[speaker_lifted_frequencies <= 5000]
speaker_lifted_magnitude = speaker_lifted_magnitude[:len(speaker_lifted_frequencies)]
anechoic_frequencies = anechoic_frequencies[anechoic_frequencies <= 5000]
anechoic_magnitude = anechoic_magnitude[:len(anechoic_frequencies)]
brooks_frequencies = brooks_frequencies[brooks_frequencies <= 5000]
brooks_magnitude = brooks_magnitude[:len(brooks_frequencies)]

plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(speaker_ground_frequencies, speaker_ground_magnitude, color='green', linewidth=0.5)
plt.title('Cross Spectrum Magnitude - Speaker on Ground')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(speaker_lifted_frequencies, speaker_lifted_magnitude, color='blue', linewidth=0.5)
plt.title('Cross Spectrum Magnitude - Speaker Lifted off Ground')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(anechoic_frequencies, anechoic_magnitude, color='red', linewidth=0.5)
plt.title('Cross Spectrum Magnitude - Anechoic w/o Speaker')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(brooks_frequencies, brooks_magnitude, color='purple', linewidth=0.5)
plt.title('Cross Spectrum Magnitude - Brooks (Ambient Noise)')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/all_cross_spectra.png')

# Plot cross spectrum phase
# Only plot up to 5000 Hz
speaker_ground_phase = speaker_ground_phase[:len(speaker_ground_frequencies)]
speaker_lifted_phase = speaker_lifted_phase[:len(speaker_lifted_frequencies)]
anechoic_phase = anechoic_phase[:len(anechoic_frequencies)]
brooks_phase = brooks_phase[:len(brooks_frequencies)]
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(speaker_ground_frequencies, speaker_ground_phase, color='green', linewidth=0.5)
plt.title('Cross Spectrum Phase - Speaker on Ground')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(speaker_lifted_frequencies, speaker_lifted_phase, color='blue', linewidth=0.5)
plt.title('Cross Spectrum Phase - Speaker Lifted off Ground')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(anechoic_frequencies, anechoic_phase, color='red', linewidth=0.5)
plt.title('Cross Spectrum Phase - Anechoic w/o Speaker')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(brooks_frequencies, brooks_phase, color='purple', linewidth=0.5)
plt.title('Cross Spectrum Phase - Brooks (Ambient Noise)')
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/all_cross_spectra_phase.png')



# Plot coherence
#Only plot up to 5000 Hz
speaker_ground_coherence_frequencies = speaker_ground_coherence_frequencies[speaker_ground_coherence_frequencies <= 5000]
speaker_ground_coherence = speaker_ground_coherence[:len(speaker_ground_coherence_frequencies)]
speaker_lifted_coherence_frequencies = speaker_lifted_coherence_frequencies[speaker_lifted_coherence_frequencies <= 5000]
speaker_lifted_coherence = speaker_lifted_coherence[:len(speaker_lifted_coherence_frequencies)]
anechoic_coherence_frequencies = anechoic_coherence_frequencies[anechoic_coherence_frequencies <= 5000]
anechoic_coherence = anechoic_coherence[:len(anechoic_coherence_frequencies)]
brooks_coherence_frequencies = brooks_coherence_frequencies[brooks_coherence_frequencies <= 5000]
brooks_coherence = brooks_coherence[:len(brooks_coherence_frequencies)]         
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(speaker_ground_coherence_frequencies, speaker_ground_coherence, color='green', linewidth=0.5)
plt.title('Coherence - Speaker on Ground')
plt.ylabel('Coherence')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(speaker_lifted_coherence_frequencies, speaker_lifted_coherence, color='blue', linewidth=0.5)
plt.title('Coherence - Speaker Lifted off Ground')
plt.ylabel('Coherence')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(anechoic_coherence_frequencies, anechoic_coherence, color='red', linewidth=0.5)
plt.title('Coherence - Anechoic w/o Speaker')
plt.ylabel('Coherence')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(brooks_coherence_frequencies, brooks_coherence, color='purple', linewidth=0.5)
plt.title('Coherence - Brooks (Ambient Noise)')
plt.ylabel('Coherence')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Cross Spectrum Final proj/Plots/all_coherence.png')

