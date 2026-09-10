import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
def list_files(path):
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
path = 'C:\Users\Sukhvansh Jain\Desktop\IIT Study Material\Semester 4\CS-671-DL\PixelSNAIL code and model\DCASE_2023_Challenge_Task_7_Submission\AudioFiles\Submissions\A\TASys02'
paths,filenames=list_files(path)
outputpath='C:\Users\Sukhvansh Jain\Desktop\IIT Study Material\Semester 4\CS-671-DL\PixelSNAIL code and model\DCASE_2023_Challenge_Task_7_Submission\AudioFiles\Submissions\A\TASys02\Train'
for i in range(len(paths)):
    file=paths[i]
    outpath=outputpath+"\\"+filenames[i][:-3]+"jpeg"
    scale,sr=librosa.load(file,sr=None)
    mel_spectogram=librosa.feature.melspectrogram(y=scale,sr=sr,n_fft=2048,hop_length=512,n_mels=40)
    log_mel_spectogram=librosa.power_to_db(mel_spectogram)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(log_mel_spectogram,x_axis='time',y_axis='mel',sr=sr)
    plt.axis('tight')
    plt.axis('off')
    plt.savefig(outpath,dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()