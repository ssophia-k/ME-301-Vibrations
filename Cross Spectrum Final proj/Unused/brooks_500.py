import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cross_spectrum(X_mag, X_phase, Y_mag, Y_phase):
    """
    Calculate the cross spectrum from the magnitude and phase of two signals.

    Parameters:
    X_mag (array): Magnitude of signal X
    X_phase (array): Phase of signal X (RADIANS)
    Y_mag (array): Magnitude of signal Y
    Y_phase (array): Phase of signal Y (RADIANS)

    Returns:
    tuple: Cross spectrum magnitude and phase
    """
    # Convert magnitude and phase to complex numbers
    X_complex = X_mag * np.exp(1j * (X_phase))
    Y_complex = Y_mag * np.exp(1j * (Y_phase))

    # Calculate cross spectrum
    cross_complex = X_complex * np.conj(Y_complex)
    cross_magnitude = np.abs(cross_complex)
    cross_phase = np.angle(cross_complex, deg=True)  # Convert to degrees
    cross_phase = np.mod(cross_phase, 360)  # Ensure phase is between 0 and 360 degrees
    cross_phase = cross_phase - 180  # Shift phase by -180 degrees

    return cross_magnitude, cross_phase


# File path
fft_mag_file_path = 'Cross Spectrum Final proj/Data/Brooks_500/fft500mag.txt'
fft_phase_file_path = 'Cross Spectrum Final proj/Data/Brooks_500/fft500pha.txt'

# Read the data, skipping the first line (header) and setting column names
fft_mag_data = pd.read_csv(fft_mag_file_path, delim_whitespace=True, skiprows=1, names=["Frequency1", "Microphone", "Frequency2", "Accelerometer"])
fft_phase_data = pd.read_csv(fft_phase_file_path, delim_whitespace=True, skiprows=1, names=["Frequency1", "Microphone", "Frequency2", "Accelerometer"])

# Combine magnitude and phase data into a single DataFrame
#cross_data = pd.merge(cross_mag_data, cross_phase_data, on="Frequency", how="inner") 

# Convert data magnitudes from dB to linear scale
fft_mag_data["Microphone"] = 10**(fft_mag_data["Microphone"] / 20)
fft_mag_data["Accelerometer"] = 10**(fft_mag_data["Accelerometer"] / 20)


# Get the power spectral density (PSD) for the microphone and accelerometer
fs = 51200 #Sampling frequency, Hz
N = len(fft_mag_data["Microphone"])
psd_microphone = (2.0 / (fs * N)) * np.abs(fft_mag_data["Microphone"])**2
psd_accelerometer = (2.0 / (fs * N)) * np.abs(fft_mag_data["Accelerometer"])**2

# Get the cross spectrum
# convert magnitude and phase to complex numbers
#microphone_complex = fft_mag_data["Microphone"] * np.exp(1j * fft_phase_data["Microphone"])
microphone_complex = fft_mag_data["Microphone"] * np.cos(fft_phase_data["Microphone"]) + 1j * fft_mag_data["Microphone"] * np.sin(fft_phase_data["Microphone"])
accelerometer_complex = fft_mag_data["Accelerometer"] * np.cos(fft_phase_data["Accelerometer"]) + 1j * fft_mag_data["Accelerometer"] * np.sin(fft_phase_data["Accelerometer"])
#accelerometer_complex = fft_mag_data["Accelerometer"] * np.exp(1j * fft_phase_data["Accelerometer"])
cross_complex = microphone_complex * np.conj(accelerometer_complex)
cross_magnitude = np.abs(cross_complex)
cross_phase = np.angle(cross_complex)
cross_phase = np.degrees(cross_phase)  # Convert to degrees
cross_phase = np.mod(cross_phase, 360)  # Ensure phase is between 0 and 360 degrees
cross_phase = cross_phase - 180  # Shift phase by -180 degrees

# Get the Power Spectral Density (PSD) for the microphone and accelerometer
psd_microphone_mag, psd_microphone_phase = cross_spectrum(fft_mag_data["Microphone"], fft_phase_data["Microphone"], fft_mag_data["Microphone"], fft_phase_data["Microphone"])
psd_accelerometer_mag, psd_accelerometer_phase = cross_spectrum(fft_mag_data["Accelerometer"], fft_phase_data["Accelerometer"], fft_mag_data["Accelerometer"], fft_phase_data["Accelerometer"])

#Get the cross sectrum
cross_spectrum_mag, cross_spectrum_phase = cross_spectrum(fft_mag_data["Microphone"], fft_phase_data["Microphone"], fft_mag_data["Accelerometer"], fft_phase_data["Accelerometer"])

# Get the coherence function
coherence = np.abs(cross_spectrum_mag)**2 / (
    psd_microphone_mag * psd_accelerometer_mag )

print(fft_mag_data.head())

# Plot cross spectrum magnitude and phase
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(fft_mag_data["Frequency1"], cross_magnitude, color='blue', linewidth=0.5)
plt.title('Cross Spectrum Magnitude')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(fft_phase_data["Frequency1"], cross_phase, color='red', linewidth=0.5)
plt.title('Cross Spectrum Phase')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot FFT data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(fft_mag_data["Frequency1"], fft_mag_data["Microphone"], label='FFT Microphone', color='green', linewidth=0.5)
plt.title('FFT Microphone')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(fft_mag_data["Frequency1"], fft_mag_data["Accelerometer"], label='FFT Accelerometer', color='red', linewidth=0.5)
plt.title('FFT Accelerometer')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot coherence function
plt.figure(figsize=(10, 6))
plt.plot(fft_mag_data["Frequency1"], coherence, color='purple', linewidth=0.5)
plt.title('Coherence Function')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.grid(True)
plt.tight_layout()
plt.show()




