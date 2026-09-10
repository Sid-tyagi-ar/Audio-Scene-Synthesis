import matplotlib.pyplot as plt
import os
import numpy as np
import librosa


def File_Path_extractor(path):
    '''
    This Function takes in the Folder path and then return the Files along with their addresses
    
    Args:
        Folder Path

    Returns:
        List containing Paths for audio file in folder and name of the file
    '''
    filenames=[]
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return
    if not os.path.isdir(path):
        print("The specified path is not a directory.")
        return
    paths=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            filenames+=[file]
            file_path = os.path.join(root, file)
            paths+=[file_path]
    return paths,filenames

def Freq_max_min_Class(path):
    """
    This function calculates the minimum and maximum frequencies present for a class of audio files.
    
    Args:
    filename: Path to the Class Folder.

    Returns:
    A tuple containing the minimum and maximum frequencies (Hz).
    """
    max_F = []
    min_F = []
    filenames=[]
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
                filenames+=[file]
                file_path = os.path.join(root, file)
                paths.append(file_path)
        for x in paths:
            y, sr = librosa.load(x,sr=None)
            spectrogram = np.abs(librosa.stft(y))

            max_freq = 0
            for freq_bin in range(spectrogram.shape[0]):
                max_val = np.max(spectrogram[freq_bin, :])
                if max_val > 0:  
                    freq = sr * freq_bin / (spectrogram.shape[1] - 1)
                    max_freq = max(max_freq, freq)
                max_F.append(max_freq)

            min_freq = np.inf    
            for freq_bin in range(spectrogram.shape[0]):
                min_val = np.amin(spectrogram[freq_bin, :][spectrogram[freq_bin, :] > 0])
                if min_val > 0:
                    freq = sr * freq_bin / (spectrogram.shape[1] - 1)
                    min_freq = min(min_freq, freq)
                min_F.append(min_freq)

    return max(max_F), min(min_F)

def custom_gammatone_filterbank(sampling_rate,freq_range,num_filters):
  
    """
    For getting ourselves a gammatone filterbank which i will input to use librosa for getting spectograms.

    Args:
    sr: Sample rate of the audio signal.
    f_range: range of frequency
    n_filters: Number of filters in the filterbank.

    Returns:
    Gammatone filterbank.
    """
    min_freq, max_freq = freq_range
    ear_q = 9.26449
    min_bw = 24.7
    order = 1
    erb_point = 21.4

    # Add a small constant to the minimum frequency to avoid division by zero
    min_freq = min_freq + 1e-6

    # Equivalent Rectangular Bandwidth (ERB) scale
    erb_filterbank = erb_point * (10 ** ((order * np.log10(max_freq / min_freq) - ear_q) / 21.4) - 1)

    center_freqs = np.linspace(min_freq, max_freq, num_filters)

    # Calculate bandwidths
    bandwidths = erb_filterbank * (10 ** ((order * np.log10(center_freqs / min_freq) - ear_q) / 21.4))

    # Create filter coefficients
    filterbank = []
    t_max = 0.1  # Maximum duration for t_vector
    t = 1.0 / sampling_rate
    t_vector_length = int(t_max * sampling_rate) + 1  # Length of t_vector
    t_vector = np.arange(0, t_max + t, t)[:t_vector_length]  # Common t_vector for all filters
    for freq, bw in zip(center_freqs, bandwidths):
        gain = np.power(t_vector, order) * np.exp(-2 * np.pi * erb_filterbank * t_vector)
        filterbank.append(gain * np.cos(2 * np.pi * freq * t_vector))

    return  np.array(filterbank)
  
def Audio2Spectograms(paths,output_path,filenames,filter):
         '''
         
         This function takes in the input wav files and corresponding outputs gammatogram 
         
         Args:
         paths: takes the file address list  
         output_path: Takes the file output destination
         filenames: takes the input file names list 
         filter: Takes in the filterbank being used 

         Returns:
         Spectograms made with custom filterbank
         '''
         for i in range(len(paths)):
            files=paths[i]
            outpath=output_path+"\\"+filenames[i][:-3]+"png"
            y,sr=librosa.load(files,sr=None)
            filtered_signals = np.zeros_like(y)
            for f in filter.T:  
                filtered_signals += np.convolve(y, f, mode='same')
            gammatograms = np.abs(librosa.stft(filtered_signals)) ** 2
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(librosa.amplitude_to_db(gammatograms, ref=np.max),
                         sr=sr, x_axis='time', y_axis='linear', cmap='viridis')
            
            plt.savefig(outpath,dpi=100, bbox_inches='tight', pad_inches=0)
            plt.axis('off')
            plt.close()

paths,filenames=File_Path_extractor(r":\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Airport")
Range =  [0,16e3]
output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\test"
filter = custom_gammatone_filterbank(num_filters=40,freq_range=Range,sampling_rate=44100)
Audio2Spectograms(paths,output_path,filenames,filter=filter)
