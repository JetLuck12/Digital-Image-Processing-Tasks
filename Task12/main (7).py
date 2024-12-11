import struct
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import os

# Функция для чтения файлов MNIST в формате idx3-ubyte
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Неверный формат файла!")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Неверный формат файла!")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Путь к файлам
images_path = "t10k-images.idx3-ubyte"
labels_path = "t10k-labels.idx1-ubyte"
model_path = "/mnt/data/mnist_model.h5"

# Проверка существования файлов
if not os.path.exists(images_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Файлы MNIST не найдены! Проверьте путь к файлам.")

# Загрузка изображений и меток
images = load_mnist_images(images_path)
labels = load_mnist_labels(labels_path)

# Нормализация данных
images = images / 255.0
images = images.reshape(images.shape[0], 28, 28, 1)

# Загрузка или создание модели MNIST
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    # Если модели нет, создаём и тренируем её
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=5, batch_size=32)
    model.save(model_path)

# Случайный выбор 10 изображений
random_indices = random.sample(range(len(images)), 10)
test_images = images[random_indices]
test_labels = labels[random_indices]

# Предсказания модели
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Вывод изображений с предсказанными и истинными метками
plt.figure(figsize=(12, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Предсказано: {predicted_labels[i]}\nИстинное: {test_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
