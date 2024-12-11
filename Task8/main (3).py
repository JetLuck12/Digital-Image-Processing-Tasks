import cv2
import numpy as np


def find_circles(filepath, resolution, distance, canny_thresh, accumulator_thresh, radius_min, radius_max):
    # Чтение изображения
    src_img = cv2.imread(filepath)
    if src_img is None:
        raise FileNotFoundError("Ошибка: не удалось загрузить изображение.")

    # Преобразование изображения в оттенки серого
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Сглаживание изображения для уменьшения шума
    blurred_img = cv2.GaussianBlur(gray_img, (9, 9), sigmaX=2)

    # Обнаружение кругов методом Хафа
    detected_circles = cv2.HoughCircles(
        blurred_img,
        method=cv2.HOUGH_GRADIENT,
        dp=resolution,
        minDist=distance,
        param1=canny_thresh,
        param2=accumulator_thresh,
        minRadius=radius_min,
        maxRadius=radius_max
    )

    # Создание копии исходного изображения для отображения результатов
    result_img = src_img.copy()
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for x, y, r in detected_circles[0, :]:
            # Рисуем окружность
            cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
            # Рисуем центр
            cv2.circle(result_img, (x, y), 2, (0, 0, 255), 3)

    return gray_img, result_img


def show_images(image_list, titles_list):
    # Вывод изображений в отдельных окнах
    for title, img in zip(titles_list, image_list):
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Настройки
image_file = 'img.png'
hough_resolution = 2.5
min_circle_distance = 30
canny_threshold = 60
hough_threshold = 80
min_circle_radius = 9
max_circle_radius = 40

# Обнаружение кругов
gray_result, circles_result = find_circles(
    image_file,
    hough_resolution,
    min_circle_distance,
    canny_threshold,
    hough_threshold,
    min_circle_radius,
    max_circle_radius
)

# Вывод результатов
show_images(
    [gray_result, circles_result],
    ["Оттенки серого", "Найденные окружности"]
)
