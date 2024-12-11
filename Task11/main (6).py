import cv2
import numpy as np

def apply_watershed_segmentation(input_image):
    """Выполняет сегментацию изображения с использованием алгоритма Watershed."""
    # Перевод в оттенки серого
    grayscale_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения
    _, binary_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфологическая обработка для очистки изображения
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    refined_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, morph_kernel, iterations=2)

    # Определение фона
    background_area = cv2.dilate(refined_mask, morph_kernel, iterations=3)

    # Расстояние до ближайших границ объектов
    distance_map = cv2.distanceTransform(refined_mask, cv2.DIST_L2, 5)
    _, foreground_area = cv2.threshold(distance_map, 0.005 * distance_map.max(), 255, cv2.THRESH_BINARY)

    # Преобразование областей в формат uint8
    foreground_area = np.uint8(foreground_area)
    boundary_region = cv2.subtract(background_area, foreground_area)

    # Нанесение меток для Watershed
    _, initial_markers = cv2.connectedComponents(foreground_area)
    initial_markers = initial_markers + 1
    initial_markers[boundary_region == 255] = 0

    # Применение Watershed
    final_markers = cv2.watershed(input_image, initial_markers)
    segmented_result = input_image.copy()
    segmented_result[final_markers == -1] = [0, 0, 255]  # Отмечаем границы красным цветом

    return segmented_result, final_markers


# Загрузка изображения
img_path = 'Lena.png'  # Укажите путь к изображению
loaded_image = cv2.imread(img_path)

if loaded_image is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")

# Выполнение сегментации
segmentation_result, watershed_markers = apply_watershed_segmentation(loaded_image)

# Отображение изображений
cv2.imshow('Original', loaded_image)
cv2.imshow('Segmented', segmentation_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
