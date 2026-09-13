from curses import meta
import os
import pandas as pd
import json
from tqdm import tqdm

AUDIOSET_PATH_DICT = {}

segment_label_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/audiocaps_segment_labels/averaged"

audiocaps_csv = [
    "test.csv",
    "train.csv",
    "val.csv"
]

audioset_path = {
    "bal_unbal_train": "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_unbal_train_data.json",
    "eval": "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_eval_data.json",
    "bal_train": "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_train_data.json",
}

def read_json(dataset_json_file):
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
    return data_json['data']

def build_audioset_metadata():
    metadata_list = []
    for k in audioset_path.keys():
        metadata_list += read_json(audioset_path[k])
    for each in metadata_list:
        wav_path = each["wav"]
        audio_basename = os.path.basename(wav_path)[:-4]
        labels = each["labels"]
        AUDIOSET_PATH_DICT[audio_basename] = [wav_path, labels]
    
def read_audiocaps_csv(csvfile):
    df = pd.read_csv(csvfile)
    data_list = []
    for row in tqdm(df.iterrows()):
        instance = {}
        file_object = row[1]
        audiocap_id = file_object["audiocap_id"]
        youtube_id = file_object["youtube_id"]
        start_time = file_object["start_time"]
        caption = file_object["caption"]
        
        try:
            wav, labels = lookup_path(youtube_id)    
        except Exception as e:
            print(e)
            continue
        
        instance["wav"]=wav
        instance["seg_label"]=os.path.join(segment_label_path, "Y%s.npy" % youtube_id)
        instance["labels"]=labels
        instance["caption"]=caption
        data_list.append(instance)
    return data_list
    
def lookup_path(youtube_id):
    metadata = AUDIOSET_PATH_DICT["Y"+youtube_id]
    return metadata[0], metadata[1]

def save_json(wav_list, fname):
    with open('datafiles/%s.json' % fname, 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)

build_audioset_metadata()

for file in audiocaps_csv:
    meta = read_audiocaps_csv(file)
    save_json(meta, fname=file)
        
import ipdb; ipdb.set_trace()

