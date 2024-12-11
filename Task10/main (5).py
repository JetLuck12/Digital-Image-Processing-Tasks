import cv2
import numpy as np


def find_and_compare_features(img1, img2):
    """Обнаруживает и сопоставляет ключевые точки на двух изображениях."""
    # Создание объекта ORB
    feature_extractor = cv2.ORB_create()

    # Обнаружение и вычисление дескрипторов
    kp1, desc1 = feature_extractor.detectAndCompute(img1, None)
    kp2, desc2 = feature_extractor.detectAndCompute(img2, None)

    # Создание матчера и поиск соответствий
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_found = matcher.match(desc1, desc2)
    matches_sorted = sorted(matches_found, key=lambda m: m.distance)

    # Визуализация первых 20 соответствий
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches_sorted[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, kp1, kp2, matches_sorted


def load_grayscale_image(filepath):
    """Загружает изображение в градациях серого."""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {filepath}")
    return image


# Пути к изображениям
img1_path = 'img.png'
img2_path = 'img_1.png'

# Загрузка изображений
image_1 = load_grayscale_image(img1_path)
image_2 = load_grayscale_image(img2_path)

# Обработка изображений
matched_img, key_pts1, key_pts2, matches = find_and_compare_features(image_1, image_2)

# Отображение результатов
cv2.imshow('Matched Features', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Вывод статистики
print(f"Число ключевых точек на изображении 1: {len(key_pts1)}")
print(f"Число ключевых точек на изображении 2: {len(key_pts2)}")
print(f"Общее количество совпадений: {len(matches)}")
