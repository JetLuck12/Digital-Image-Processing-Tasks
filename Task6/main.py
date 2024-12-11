import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_uniform_noise(img, low, high):
    random_noise = np.random.uniform(low, high, img.shape).astype(np.float32)
    result = cv2.add(img.astype(np.float32), random_noise)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_gaussian_noise(img, avg, deviation):
    gaussian_random = np.random.normal(avg, deviation, img.shape).astype(np.float32)
    result = cv2.add(img.astype(np.float32), gaussian_random)
    return np.clip(result, 0, 255).astype(np.uint8)

# Загружаем изображение
path_to_image = 'Lena.png'  # Укажите путь к файлу
input_image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

if input_image is None:
    raise ValueError("Не удалось загрузить изображение. Проверьте путь.")

# Наложение шумов
image_with_uniform_noise = apply_uniform_noise(input_image, -50, 50)
image_with_gaussian_noise = apply_gaussian_noise(input_image, 0, 25)

# Обработка изображений фильтрами
smoothed_gaussian = cv2.GaussianBlur(image_with_gaussian_noise, (5, 5), 0)
smoothed_median = cv2.medianBlur(image_with_gaussian_noise, 5)

# Создание фильтра и применение
filter_matrix = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]], dtype=np.float32)
custom_filtered_image = cv2.filter2D(input_image, -1, filter_matrix)

# Вычисление Собеля
sobel_horizontal = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_vertical = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_horizontal, sobel_vertical)

# Лапласиан
laplace_filtered = cv2.Laplacian(input_image, cv2.CV_64F, ksize=3)

# Построение графиков
plt.figure(figsize=(12, 10))

images = [
    (input_image, 'Исходное изображение'),
    (image_with_uniform_noise, 'Шум равномерного распределения'),
    (image_with_gaussian_noise, 'Шум нормального распределения'),
    (smoothed_gaussian, 'Гауссовое сглаживание'),
    (smoothed_median, 'Медианное сглаживание'),
    (custom_filtered_image, 'Произвольный фильтр'),
    (sobel_combined, 'Градиент Собеля'),
    (laplace_filtered, 'Фильтр Лапласа')
]

for i, (img, title) in enumerate(images, start=1):
    plt.subplot(3, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
