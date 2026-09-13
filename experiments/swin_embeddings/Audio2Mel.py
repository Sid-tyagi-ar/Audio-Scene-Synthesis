import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


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

def Audio2Mel(Length,output_path,filenames):
    '''
    This Function takes in the Number of Files in the specific Folder and then return the Mel-Spectogram of the respective audios
    
    Args:
        Length of the List containing Paths of wav File, Output destination 

    Returns:
        Folder Created containing Mel spectogram of the wav files 
    '''
    for i in range(Length):
            file=paths[i]
            outpath=output_path+"\\"+filenames[i][:-3]+"png"
            scale,sr=librosa.load(file,sr=None)
            mel_spectogram=librosa.feature.melspectrogram(y=scale,sr=sr,n_fft=2048,hop_length=512,n_mels=40)
            log_mel_spectogram=librosa.power_to_db(mel_spectogram)
            plt.figure(figsize=(25,10))
            librosa.display.specshow(log_mel_spectogram,x_axis='time',y_axis='mel',sr=sr)
            plt.axis('tight')
            plt.axis('off')
            plt.savefig(outpath,dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()


# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Airport")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Airport"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Bus")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Bus"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Metro")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Metro"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Metro_Station")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Metro_Station"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Park")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Park"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Public_Square")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Public_Square"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Shopping_Mall")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Shopping_Mall"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Street_Pedestrian")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Street_Pedestrian"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
# paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Street_Traffic")
# output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Street_Traffic"
# length = len(paths)
# Audio2Mel(length,output_path,filenames)
paths,filenames=File_Path_extractor(r"G:\workspace\soundscene\Tram")
output_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Tram"
length = len(paths)
Audio2Mel(length,output_path,filenames)