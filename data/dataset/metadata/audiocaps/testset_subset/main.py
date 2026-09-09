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
new_metadata = {}
new_metadata["data"] = []
already_have = []

for each in metadata["data"]:
    if(os.path.basename(each["wav"]) in already_have):
        continue
    else:
        already_have.append(os.path.basename(each["wav"]))
        new_metadata["data"].append(each)

write_json(new_metadata, "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat.json")
import ipdb; ipdb.set_trace()