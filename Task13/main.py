import numpy as np
import cv2
from PIL import Image

# Function to create a depth map from an image
def create_depth_map_from_image(image_path, image_size=(500, 300)):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Resize the image to fit the depth map size
    img_resized = img.resize((image_size[1], image_size[0]))  # Resize to (width, height)

    # Convert the image to a NumPy array
    depth_map = np.array(img_resized, dtype=np.uint8)

    return depth_map

# Function to generate the autostereogram
def generate_autostereogram(depth_map, pattern_size=10):
    height, width = depth_map.shape

    # Random pattern
    pattern = np.random.randint(0, 256, (height, pattern_size), dtype=np.uint8)

    # Create autostereogram
    autostereogram = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Calculate disparity based on depth map
            disparity = depth_map[y, x] // 10  # Scale depth value for disparity

            if x - disparity >= 0:
                autostereogram[y, x] = autostereogram[y, x - disparity]
            else:
                autostereogram[y, x] = pattern[y, x % pattern_size]

    return autostereogram

# Main program
if __name__ == "__main__":
    # Path to the input image
    image_path = "Koala.png"  # Replace with your image path

    # Create a depth map from the image
    depth_map = create_depth_map_from_image(image_path, image_size=(300, 400))

    # Generate the autostereogram
    autostereogram = generate_autostereogram(depth_map)

    # Save and display the results
    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Autostereogram", autostereogram)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
