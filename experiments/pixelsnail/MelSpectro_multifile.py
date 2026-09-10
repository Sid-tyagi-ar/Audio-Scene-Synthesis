import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_output_subfolders(output_path, subfolders):
    """Creates output subfolders if they don't already exist."""
    for subfolder in subfolders:
        output_subfolder = os.path.join(output_path, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

def process_audio_files(input_path, output_path):
    """Processes audio files within a single subfolder."""
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                output_subfolder = root.replace(input_path, output_path)  # Preserve subclass structure
                outpath = os.path.join(output_subfolder, file[:-3] + "jpeg")

                try:
                    scale, sr = librosa.load(file_path, sr=None)
                    mel_spectogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
                    log_mel_spectogram = librosa.power_to_db(mel_spectogram)

                    plt.figure(figsize=(25, 10))
                    librosa.display.specshow(log_mel_spectogram, x_axis='time', y_axis='mel', sr=sr)
                    plt.axis('tight')
                    plt.axis('off')
                    plt.savefig(outpath, dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close()
                except Exception as e:
                    print(f"Error processing file: {file_path}. Error: {e}")

# Input and output paths
input_path = r'C:\Users\Sukhvansh Jain\Desktop\IIT Study Material\Semester 4\CS-671-DL\PixelSNAIL code and model\DCASE_2023_Challenge_Task_7_Submission\AudioFiles\Submissions\A\TASys03'
output_path = r'C:\Users\Sukhvansh Jain\Desktop\IIT Study Material\Semester 4\CS-671-DL\PixelSNAIL code and model\DCASE_2023_Challenge_Task_7_Submission\AudioFiles\Submissions\A\TASys03\Train'

# Iterate through subfolders and process files
subfolders = [os.path.basename(path) for path in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, path))]
create_output_subfolders(output_path, subfolders)

for subfolder in subfolders:
    input_subfolder = os.path.join(input_path, subfolder)
    output_subfolder = os.path.join(output_path, subfolder)
    process_audio_files(input_subfolder, output_subfolder)
