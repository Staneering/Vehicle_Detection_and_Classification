import os
import cv2
import random

def load_images_from_subdirectories(directory):
    image_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

# def load_and_sample_images(directory, sample_size=300):
#     image_files = [os.path.join(directory, f) for f in os.listdir(directory)
#                    if os.path.isfile(os.path.join(directory, f))]
#     if len(image_files) > sample_size:
#         return random.sample(image_files, sample_size)
#     return image_files

def load_and_sample_images(directory, sample_size=300):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))
    if len(image_files) > sample_size:
        return random.sample(image_files, sample_size)
    return image_files
