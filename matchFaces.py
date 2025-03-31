import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import lu

# Load images in grayscale mode
image1 = cv2.imread("sample1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("sample5.jpg", cv2.IMREAD_GRAYSCALE)

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

# Display edges
plt.subplot(1, 2, 1), plt.imshow(edges1, cmap="gray"), plt.title("Face 1 Edges")
plt.subplot(1, 2, 2), plt.imshow(edges2, cmap="gray"), plt.title("Face 2 Edges")
plt.show()

def extract_landmarks(image, num_points=40):
    """Extracts landmark points from an edge-detected image using contour detection."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    landmark_points = []
    for contour in contours:
        for point in contour:#since point is a 2d array
            x, y = point[0]
            landmark_points.append((x, y))
    
    # Sort and select top 'num_points'
    landmark_points = sorted(landmark_points, key=lambda p: (p[1], p[0]))[:num_points]
    
    return np.array(landmark_points)

# Extract landmark points for both images
landmarks1 = extract_landmarks(edges1)
landmarks2 = extract_landmarks(edges2)

# Ensure both have the same number of points
num_points = min(len(landmarks1), len(landmarks2))
landmarks1 = landmarks1[:num_points]
landmarks2 = landmarks2[:num_points]

# Convert landmark points into NumPy arrays
X1 = np.array([p[0] for p in landmarks1])
Y1 = np.array([p[1] for p in landmarks1])

X2 = np.array([p[0] for p in landmarks2])
Y2 = np.array([p[1] for p in landmarks2])

# ✅ Fix duplicate X-values issue for interpolation
unique_X1, unique_indices1 = np.unique(X1, return_index=True)
Y1 = Y1[unique_indices1]  # Keep corresponding Y values

unique_X2, unique_indices2 = np.unique(X2, return_index=True)
Y2 = Y2[unique_indices2]  # Keep corresponding Y values

# Sort (X, Y) pairs based on X values
sorted_indices1 = np.argsort(unique_X1)
X1 = unique_X1[sorted_indices1]
Y1 = Y1[sorted_indices1]

sorted_indices2 = np.argsort(unique_X2)
X2 = unique_X2[sorted_indices2]
Y2 = Y2[sorted_indices2]

# Create interpolation functions for both faces
f1 = interp1d(X1, Y1, kind="linear", fill_value="extrapolate")
f2 = interp1d(X2, Y2, kind="linear", fill_value="extrapolate")

# Generate interpolated points
X_new = np.linspace(min(X1.min(), X2.min()), max(X1.max(), X2.max()), 100)
Y1_interp = f1(X_new)
Y2_interp = f2(X_new)

# Compute Mean Squared Error (MSE) between the interpolated curves
mse = np.mean((Y1_interp - Y2_interp) ** 2)

# Convert landmarks into matrix form
A1 = np.column_stack([X1, Y1, np.ones(len(X1))])
A2 = np.column_stack([X2, Y2, np.ones(len(X2))])

# Perform LU Decomposition 
P1, L1, U1 = lu(A1)
P2, L2, U2 = lu(A2)
#PA = LU
#In L diag entries are 1
# Compute determinant of U (Face Structure Strength)
det1 = np.abs(np.linalg.det(U1))
det2 = np.abs(np.linalg.det(U2))

# Compute similarity ratio using determinant comparison
lu_similarity =0
if max(det1, det2) == 0:  # Avoid division by zero
    lu_similarity = 100  # If both determinants are zero, assume perfect similarity
else:
    lu_similarity = 100 - (abs(det1 - det2) / max(det1, det2)) * 100




def power_method(A, num_iter=10):
    """Computes dominant eigenvalue using the Power Method."""
    b = np.ones(A.shape[1])  # Now, b matches A's columns (3 elements)
    for _ in range(num_iter):
        b = np.dot(A.T @ A, b)  # Use A.T @ A to get a square matrix
        b = b / np.linalg.norm(b)  # Normalize
    return b

# Compute eigenvalue similarity between faces
eig1 = power_method(A1)
eig2 = power_method(A2)
eig_similarity = 100 - (np.linalg.norm(eig1 - eig2) / max(np.linalg.norm(eig1), np.linalg.norm(eig2))) * 100

# Compute final face similarity score
final_similarity = (lu_similarity + eig_similarity - mse) / 2
final_similarity = max(0, min(100, final_similarity))  # Ensure score is between 0-100

# Display result
print(f"Face Similarity Score: {final_similarity:.2f}%")

if final_similarity > 70:
    print("Faces Match ✅")
else:
    print("Faces Do Not Match ❌")
