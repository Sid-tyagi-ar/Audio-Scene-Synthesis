import os
import json

def write_json(my_dict, fname):
    # print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data

filepath = "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/datafiles/audiocaps_test_label.json"

metadata = load_json(filepath)

already_have = []

current_meta = {}

for each in metadata["data"]:
    if(os.path.basename(each["wav"]) not in current_meta.keys()):
        current_meta[os.path.basename(each["wav"])] = []
    current_meta[os.path.basename(each["wav"])].append(each)

for i in range(5):
    new_metadata = {}
    new_metadata["data"] = []
    for k in current_meta.keys():
        new_metadata["data"].append(current_meta[k][i])
    write_json(new_metadata, "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat_subset_%s.json" % i)
import ipdb; ipdb.set_trace()