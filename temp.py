import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images in grayscale mode
image1 = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("face2.jpg", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both images not found!")
    exit()

# Resize images to the same dimensions
image1 = cv2.resize(image1, (200, 200))
image2 = cv2.resize(image2, (200, 200))

# Apply edge detection using Canny algorithm
edges1 = cv2.Canny(image1, 50, 150)
edges2 = cv2.Canny(image2, 50, 150)
plt.imshow(edges1, cmap="gray")
plt.title("Edge-detected Image")
plt.show()

# Function to extract landmark points
def extract_landmarks(image, num_points=20):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmark_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            landmark_points.append((x, y))
    # Sort and select top 'num_points'
    landmark_points = sorted(landmark_points, key=lambda p: (p[1], p[0]))[:num_points]
    return np.array(landmark_points)

# Extract landmarks for both images
landmarks1 = extract_landmarks(edges1)
landmarks2 = extract_landmarks(edges2)

# Ensure both have the same number of points
num_points = min(len(landmarks1), len(landmarks2))
landmarks1 = landmarks1[:num_points]
landmarks2 = landmarks2[:num_points]

# Convert grayscale images to BGR for visualization
image1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
image2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

# Draw landmarks on images
for (x, y) in landmarks1:
    cv2.circle(image1_color, (x, y), 3, (0, 255, 0), -1)  # Green dots for landmarks

for (x, y) in landmarks2:
    cv2.circle(image2_color, (x, y), 3, (0, 255, 0), -1)  # Green dots for landmarks

# Display the images with landmarks
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1_color, cv2.COLOR_BGR2RGB))
plt.title("Face 1 with Landmarks")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image2_color, cv2.COLOR_BGR2RGB))
plt.title("Face 2 with Landmarks")
plt.axis("off")

plt.show()
