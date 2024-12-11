import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'cheneeses.jpg'
template_path = 'chineese2.jpg'

image = cv2.imread(image_path)
template = cv2.imread(template_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)


def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


def adjust_contrast(image, value):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)


def add_gaussian_noise(image, mean=0, std=1):
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, gaussian_noise)


def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height))


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))


def apply_perspective_transform(image):
    height, width = image.shape[:2]
    pts1 = np.float32([[100, 100], [width - 100, 100], [100, height - 100], [width - 100, height - 100]])

    pts2 = np.float32([[300, 300], [width - 50, 70], [50, height - 50], [width - 50, height - 70]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(image, matrix, (width, height))


transformations = {
    'Original': image,
    'Brighter': adjust_brightness(image, 50),
    'Higher Contrast': adjust_contrast(image, 2),
    'Noisy': add_gaussian_noise(image),
    'Rotated': rotate_image(image, 45),
    'Perspective': apply_perspective_transform(image)
}


for key, transformed_image in transformations.items():
    gray_transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_transformed_image, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(transformed_image, top_left, bottom_right, (0, 255, 0), 2)

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Object in {key}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title("Template")
    plt.axis('off')

    plt.show()
