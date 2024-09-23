import cv2
from PIL import Image
import os

# Function to crop and resize images
def crop_and_resize(image_path, output_path1, output_path2, crop_area, resize_dim):
    # Load the image
    img = Image.open(image_path)
    
    # Crop the image (crop_area = (left, top, right, bottom))
    cropped_img = img.crop(crop_area)
    
    # Resize the image to the desired dimension (resize_dim = (width, height))
    resized_img = cropped_img.resize(resize_dim)
    
    # Save the resized image
    resized_img.save(output_path1)
    resized_img.save(output_path2)

# Example usage, change the name for each player shooting or goalkeeping
# annotation ([1 or 0] successful shooting/goalkeeping?)_(first initial_last name)_(file number)
image_path = r"D:\Soccer_dataset\Test\1_p_dybala_001.png"
output_path1 = r"D:\Soccer_dataset\Test_Normalized\0_k_coman_001.png"
output_path2 = r"D:\Soccer_dataset\Test_Normalized\0_h_lloris_002.png"

# Define the area to crop (these values are just an example and should be adjusted)
# For example, to crop out a bar at the bottom or top, adjust the dimensions accordingly
left = 500
top = 200
right = 1400
bottom = 800
crop_area = (left, top, right, bottom)  # Adjust based on your image size and what you want to crop out
resize_dim = (500,500)  # Resize to 224x224 pixels

# Run the crop and resize function
crop_and_resize(image_path, output_path1, output_path2,crop_area, resize_dim)