from PIL import Image
import os

def resize_with_transparent_padding(paths, Length, output_folder, target_size=(1200, 1200)):
    for i in range(Length):
        file = paths[i]
        with Image.open(file) as image:
            # Get original image dimensions
            width, height = image.size

            # Calculate new dimensions to maintain aspect ratio
            new_width, new_height = target_size
            if width > height:
                # Wider image, adjust height based on width
                new_height = int(height * (new_width / width))
            else:
                # Taller image, adjust width based on height
                new_width = int(width * (new_height / height))

            # Create a new transparent image with target size filled with white (optional)
            background = Image.new("RGBA", target_size, (255, 255, 255, 0))  # Change to (0, 0, 0, 0) for black background

            # Paste the resized image onto the background at the center
            offset_x = (target_size[0] - new_width) // 2
            offset_y = (target_size[1] - new_height) // 2
            background.paste(image.resize((new_width, new_height)), (offset_x, offset_y))

            # Save the final image with transparent background
            filename = os.path.basename(file)
            output_path = os.path.join(output_folder, filename)
            background.save(output_path)
            print(f"Image resized with aspect ratio preserved and saved to {output_path}")


def file_path_extractor(folder_path):
    '''
    This Function takes in the Folder path and then return the Files along with their addresses
    
    Args:
        Folder Path

    Returns:
        List containing Paths for  file in folder and name of the file
    '''
    filenames = []
    if not os.path.exists(folder_path):
        print("The specified path does not exist.")
        return
    if not os.path.isdir(folder_path):
        print("The specified path is not a directory.")
        return
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(file)
            file_path = os.path.join(root, file)
            paths.append(file_path)
    return paths, filenames

p1, f1 = file_path_extractor(r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Tram")   
output_folder = r"C:\Users\Sid1t\OneDrive\Documents\Sound scene synthesis\Data_Root\Tram"
resize_with_transparent_padding(p1, len(p1), output_folder)
