import os
import random

random.seed(69)

def split_data(files_list):
    # Shuffle the files list
    random.shuffle(files_list)
    
    # Calculate the number of files for each split (70-20-10 ratio)
    total_files = len(files_list)
    train_size = 900
    val_size = 200
    test_size = 100
    # Divide the files into Training, Validation, and Test sets
    train_set = files_list[:train_size]
    val_set = files_list[train_size:train_size+val_size]
    test_set = files_list[train_size+val_size:train_size+val_size+test_size]
    
    return train_set, val_set, test_set

def get_files_list(folder_path):
    # Initialize an empty list to store file paths
    files_list = []
    
    # Traverse through the folder and its subfolders to get all files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            # Append the file path to the list
            files_list.append(file_path)
    
    return files_list

folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Airport"
files_list = get_files_list(folder_path)
train_set, val_set, test_set = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Bus"
files_list = get_files_list(folder_path)
train_set1, val_set1, test_set1 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Metro"
files_list = get_files_list(folder_path)
train_set2, val_set2, test_set2 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Metro_Station"
files_list = get_files_list(folder_path)
train_set3, val_set3, test_set3 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Park"
files_list = get_files_list(folder_path)
train_set4, val_set4, test_set4 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Public_Square"
files_list = get_files_list(folder_path)
train_set5, val_set5, test_set5 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Shopping_Mall"
files_list = get_files_list(folder_path)
train_set6, val_set6, test_set6 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Street_Pedestrian"
files_list = get_files_list(folder_path)
train_set7, val_set7, test_set7 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Street_Traffic"
files_list = get_files_list(folder_path)
train_set8, val_set8, test_set8 = split_data(files_list)
folder_path = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Tram"
files_list = get_files_list(folder_path)
train_set9, val_set9, test_set9 = split_data(files_list)

def write_to_file(file_paths, file_name,i):
    with open(file_name, 'a') as file:
        for path in file_paths:
            file.write(path + ' '+ str(i) + '\n')

train_file = r'C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\train.txt'
val_file = r'C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\val.txt'
test_file = r'C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\test.txt'
print(len(train_set))
train = [train_set,train_set1,train_set2,train_set3,train_set4,train_set5,train_set6,train_set7,train_set8,train_set9]
val = [val_set,val_set1,val_set2,val_set3,val_set4,val_set5,val_set6,val_set7,val_set8,val_set9]
test = [test_set,test_set1,test_set2,test_set3,test_set4,test_set5,test_set6,test_set7,test_set8,test_set9]
for i in range(10):
    write_to_file(train[i], train_file,i)
    write_to_file(val[i], val_file,i)
    write_to_file(test[i], test_file,i)

print("Files written successfully.")
