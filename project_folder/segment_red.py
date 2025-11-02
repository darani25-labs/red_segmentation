import cv2
import numpy as np

# Load the image
# NOTE: Replace 'festival_image.jpg' with the actual path to your festival image
image = cv2.imread('images/festival_image.jpg')

if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# --- Define two HSV ranges for red color ---
# In OpenCV, Hue (H) range is [0, 179], Saturation (S) range is [0, 255], Value (V) range is [0, 255].
# Red color is split between two ranges: (0-10) and (170-179).
# Saturation and Value are typically kept high to avoid gray/white/black.

# 1. Lower range for red (0 to 10)
# Adjust the S and V minimum values (e.g., 50 and 50) based on your image lighting
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

# 2. Upper range for red (170 to 179)
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([179, 255, 255])

# --- Create the Masks ---

# Threshold the HSV image to get the first red mask
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

# Threshold the HSV image to get the second red mask
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine the two masks using a bitwise OR operation to get the final red mask
final_mask = mask1 + mask2 
# Alternatively: final_mask = cv2.bitwise_or(mask1, mask2)

# --- Apply the Mask to the Original Image (Optional) ---

# Apply the mask to the original image using a bitwise AND operation
# This highlights only the red objects in the original color
segmented_output = cv2.bitwise_and(image, image, mask=final_mask)

# --- Display Results ---

cv2.imshow('Original Image', image)
cv2.imshow('Red Mask (Output)', final_mask) # The binary mask is your requested output
cv2.imshow('Segmented Red Objects', segmented_output)

# Wait for a key press and then close all display windows
cv2.waitKey(0)
cv2.destroyAllWindows()