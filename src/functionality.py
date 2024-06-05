import os

import numpy as np
import cv2
from .config import (
    parameters,
)



def post_process_fractions_plot(fractions, iron):
    mean_iron = 0
    if iron:    
        mean_iron = np.mean(iron)
    areas_combined = [area for sublist in fractions for area in sublist]
    combinded_np = np.concatenate(areas_combined)
    areas_sorted = np.sort(combinded_np)
    areas_cdf = np.arange(1, len(areas_sorted)+1) / len(areas_sorted)
    return {
        "sorted": areas_sorted,
        "cdf": areas_cdf,
        "mean_iron": mean_iron
    }

def image_fractions_preproc(image, k):
    """
    Necessary conversions
    """
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    return apply_filters(gray_image, k)

def image_preproc(image, k):
    """
    Necessary conversions
    """
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Create a binary mask for the black artifacts (assuming artifacts are pure black)
    _, artifact_mask = cv2.threshold(gray_image, 35, 255, cv2.THRESH_BINARY_INV)

    # Dilate the mask to cover the edges
    kernel = np.ones((5, 5), np.uint8)
    artifact_mask_dilated = cv2.dilate(artifact_mask, kernel, iterations=1)
        
    # Inpaint the artifacts using the mask
    inpainted_image = cv2.inpaint(gray_image, artifact_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    if parameters["show_images"]:
        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Inpainted Image', inpainted_image)
        cv2.imshow('Artifact Mask', artifact_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return apply_filters(inpainted_image, k)

def apply_filters(image, k):

    # Blurring images with Gaussian filter - adjust Kernel and sigmaX size to change the effect
    # blurred_image = cv2.GaussianBlur(image, (k, k), 3)

    median = cv2.medianBlur(image, k)

    if parameters["show_images"]:
        dimensions = (1200, 800)
        resized_image = cv2.resize(median, dimensions)
        cv2.imshow('blurred', resized_image)
        cv2.waitKey(0)

    return median


def load_images(folder_path):
    images = []
    supported_formats = ['.png', '.jpg', '.tif', '.png']

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_formats):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, filename))

    return images

