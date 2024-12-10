import cv2
import numpy as np
import os
import pandas as pd
from fastapi import FastAPI

CONTRAST_THRESHOLD = 25
df = pd.DataFrame()

app=FastAPI()



def show_image(image_path, screen_width=1080, screen_height=920):
    """
    Display an image fitted to the screen size while maintaining the aspect ratio.
    
    :param image_path: Path to the image file
    :param screen_width: Width of the screen (default 800)
    :param screen_height: Height of the screen (default 600)
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Image not found or could not be loaded.")
        return
    
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]
    
    # Calculate scaling factors for width and height
    scale_width = screen_width / original_width
    scale_height = screen_height / original_height
    
    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_width, scale_height)
    
    # Calculate the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Display the resized image
    cv2.imshow("Image", resized_image)
    
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    
    # Destroy the window after keypress
    cv2.destroyAllWindows()


def find_jpg_files(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                yield os.path.join(root, file)

def calculate_contrast(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found or invalid image format.")
        return None
    
    # Calculate the mean and standard deviation of pixel intensities
    mean, stddev = cv2.meanStdDev(image)
    
    # Contrast is often assessed using the standard deviation of pixel intensities
    contrast = stddev[0][0]
    
    return contrast

c_ls = []
img_ls = []
directory_path = 'C:\mukil\dev\VideoAnalytics\ikshana\snapshots\snapshots'
for jpg_file in find_jpg_files(directory_path):
    image_path = jpg_file
    contrast_value = calculate_contrast(image_path)
    if contrast_value is not None:
        # print(f"The contrast of the image is: {contrast_value}")
        c_ls.append(contrast_value)
        img_ls.append(jpg_file)
        
min_val = min(c_ls)
print(min_val)
min_val_index = c_ls.index(min_val)
print(img_ls[min_val_index])

res_ls = []
for val in c_ls:
    if val<CONTRAST_THRESHOLD:
        res_ls.append(val)
res_img_ls = []
for val in res_ls:
    idx = c_ls.index(val)
    img_path = img_ls[idx]
    img_id = img_path.split('\\')[-1]
    res_img_ls.append(img_id)
    print(val)
    show_image(img_path)
print(res_img_ls)




