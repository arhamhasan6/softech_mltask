

# import json
# import os

# def json_to_yolo(json_file, yolo_output_dir, class_mapping):
#     with open(json_file, 'r') as json_file:
#         data = json.load(json_file)

#     for entry in data['annotations']:
#         category_id = entry['category_id']
#         class_name = class_mapping.get(category_id, None)
#         if class_name is None:
#             continue

#         x, y, width, height = entry['bbox']
#         image_id = entry['image_id']  # No need to convert image_id to a string

#         # Create YOLO annotation line
#         center_x = (x + width / 2) / 1920  # Assuming image width is 1920 (you can change this)
#         center_y = (y + height / 2) / 1080  # Assuming image height is 1080 (you can change this)
#         yolo_width = width / 1920
#         yolo_height = height / 1080

#         yolo_line = f"{class_name} {center_x:.6f} {center_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n"

#         # Get the original image file name (assuming it's available in the same directory as the JSON file)
#         image_name = f"{image_id}.jpg"

#         # Save the YOLO annotation in the output directory with the original image name
#         yolo_output_file = os.path.join(yolo_output_dir, os.path.splitext(image_name)[0] + ".txt")

#         os.makedirs(yolo_output_dir, exist_ok=True)

#         with open(yolo_output_file, 'a') as yolo_file:
#             yolo_file.write(yolo_line)

# if __name__ == "__main__":
#     # JSON file with annotation data
#     json_file = r"C:\Users\arham.hasan\Downloads\archive\val\COCO_val_annos.json"
#     # Output directory for YOLO annotation files
#     yolo_output_dir = "yolo_annotations"

#     # Class mapping from category_id to class names (modify as needed)
#     class_mapping = {
#         1: 0,
       
#         # Add more class mappings as needed
#     }

#     # Create the output directory if it doesn't exist
#     os.makedirs(yolo_output_dir, exist_ok=True)

#     json_to_yolo(json_file, yolo_output_dir, class_mapping)


import os
import glob

# Specify the directory containing the images
image_dir = r'C:\Users\arham.hasan\Downloads\archive\val\images'

# Use glob to get a list of image files in the directory
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

# Sort the image files by their names
sorted_image_files = sorted(image_files)




l=[]


# Now, you can iterate through the sorted image files
for image_file in sorted_image_files:
    # Do something with each image file
    length = image_file
    l.append(int(length[50:-4]))

l.sort()
print(len(l))


# Specify the directory containing the text files
folder_path = r'C:\Users\arham.hasan\Downloads\JSON2YOLO-master\JSON2YOLO-master\yolo_annotations'
new=  r'C:\Users\arham.hasan\Downloads\JSON2YOLO-master\JSON2YOLO-master\yolo_val'
# Define a list of new names corresponding to the files in the folder



# Get a list of all text files in the folder
text_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".txt")]

# Sort the list of filenames
sorted_text_files = sorted(text_files)
tx=[]

for i in sorted_text_files:
    tx.append(int(i[:-4]))
tx.sort()
print(len(tx))







for i in range(len(tx)):
        old_file_path = os.path.join(folder_path, str(tx[i])+".txt")
       
        # Create the full path for the new file
        new_file_path = os.path.join(new, str(l[i])+".txt")

        print(old_file_path, new_file_path)
        os.rename(old_file_path, new_file_path)

        # print(f"Renamed: {str(tx[i])} to {str(l[i])}")
        # print()
      

 

