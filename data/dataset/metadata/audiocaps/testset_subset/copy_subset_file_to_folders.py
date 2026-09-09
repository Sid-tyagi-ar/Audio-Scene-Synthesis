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
    
for i in range(5):
    target_folder = "/mnt/fast/nobackup/users/hl01486/datasets/audiocaps_test_subset/%s" % i
    metadata = "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat_subset_%s.json" % i
    os.makedirs(target_folder, exist_ok=True)
    meta = load_json(metadata)["data"]
    
    for each in meta:
        filepath = each["wav"]
        cmd = "cp %s %s" % (filepath, target_folder)
        # os.system(cmd)