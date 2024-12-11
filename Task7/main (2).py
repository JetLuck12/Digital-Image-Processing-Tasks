import cv2
import numpy as np


def analyze_image(file_path, threshold1, threshold2):
    # Чтение изображения
    src_image = cv2.imread(file_path)
    if src_image is None:
        raise FileNotFoundError("Ошибка: файл изображения не найден или поврежден.")

    # Преобразование в оттенки серого
    grayscale = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    # Выявление границ методом Кэнни
    edge_map = cv2.Canny(grayscale, threshold1, threshold2)

    # Поиск контуров
    found_contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображение контуров на исходном изображении
    outlined_image = src_image.copy()
    cv2.drawContours(outlined_image, found_contours, -1, (0, 255, 0), thickness=2)

    return grayscale, edge_map, outlined_image


def show_results(image_list, window_titles):
    for window_name, image in zip(window_titles, image_list):
        cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Основные настройки
path_to_file = 'img.png'
canny_t1 = 100
canny_t2 = 200

# Обработка изображения
gray, edges_detected, image_with_contours = analyze_image(path_to_file, canny_t1, canny_t2)

# Вывод результатов
show_results(
    [gray, edges_detected, image_with_contours],
    ["Серый масштаб", "Края (Canny)", "Изображение с контурами"]
)
