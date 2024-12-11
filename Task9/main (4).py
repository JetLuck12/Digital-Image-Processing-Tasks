import cv2
import numpy as np


def modify_brightness(img, adjustment):
    """Изменяет яркость изображения."""
    return cv2.convertScaleAbs(img, alpha=1, beta=adjustment)


def detect_harris_corners(img, block, aperture, sensitivity):
    """Обнаружение углов методом Харриса."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    response = cv2.cornerHarris(gray_img, block, aperture, sensitivity)
    response = cv2.dilate(response, None)
    result_img = img.copy()
    result_img[response > 0.01 * response.max()] = [0, 0, 255]
    return result_img


def detect_shi_tomasi_corners(img, max_points, quality, dist):
    """Обнаружение углов методом Ши-Томаси."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_points = cv2.goodFeaturesToTrack(gray_img, max_points, quality, dist)
    detected_points = np.int32(detected_points)
    result_img = img.copy()
    for point in detected_points:
        x, y = point.ravel()
        cv2.circle(result_img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    return result_img


def perform_geometric_transforms(img):
    """Применение аффинных и перспективных преобразований."""
    height, width, _ = img.shape

    # Аффинное преобразование
    src_points_affine = np.float32([[50, 50], [200, 50], [50, 200]])
    dst_points_affine = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv2.getAffineTransform(src_points_affine, dst_points_affine)
    affine_transformed = cv2.warpAffine(img, affine_matrix, (width, height))

    # Перспективное преобразование
    src_points_perspective = np.float32([[50, 50], [400, 50], [50, 400], [100, 400]])
    dst_points_perspective = np.float32([[10, 100], [400, 50], [50, 400], [190, 290]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points_perspective, dst_points_perspective)
    perspective_transformed = cv2.warpPerspective(img, perspective_matrix, (width, height))

    return affine_transformed, perspective_transformed


# Загрузка изображения
image_path = 'img_1.png'
input_image = cv2.imread(image_path)
if input_image is None:
    raise FileNotFoundError("Ошибка: не удалось загрузить изображение.")

# Обработка изображения
affine_img, perspective_img = perform_geometric_transforms(input_image)
brighter_img = modify_brightness(input_image, adjustment=50)
darker_img = modify_brightness(input_image, adjustment=-50)

# Анализ изображения
results = {
    "Perspective - Harris": detect_harris_corners(perspective_img, block=2, aperture=3, sensitivity=0.04),
    "Perspective - Shi-Tomasi": detect_shi_tomasi_corners(perspective_img, max_points=100, quality=0.01, dist=10)
}

# Отображение результатов
for title, img in results.items():
    cv2.imshow(title, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
