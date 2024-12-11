from PIL import Image
from random import randint

# Загрузка фонового изображения
bg = Image.open("bg.png")
skip = bg.size[0]
m = Image.open("depth_map.png").convert("RGB")
rbg = Image.new("RGB", m.size)  # Убедитесь, что изображение в режиме RGB

# Создаем плиточный фон
for x in range(m.size[0] // bg.size[0] + 1):
    for y in range(m.size[1] // bg.size[1] + 1):
        rbg.paste(bg, (x * bg.size[0], y * bg.size[1]))
bg = rbg
out = rbg.copy()  # Копируем фон для результата

# Генерация автостереограммы
for y in range(m.size[1]):
    data = {}
    for x in range(m.size[0]):
        letter = randint(0, 255), randint(0, 255), randint(0, 255)  # Случайный RGB-цвет
        if x > skip:
            s = m.getpixel((x, y))
            s = s[0]  # Используем только первый канал (R)
            s = skip - skip * s / 256
        else:
            s = 0
        s += skip
        s = x - s
        if s < 0:
            ss = letter  # Случайный цвет, если индекс выходит за границы
        else:
            ss = data[int(s)]  # Цвет, смещенный по карте глубины
        data[x] = ss
        out.putpixel((x, y), tuple(ss))  # Убедитесь, что формат RGB сохраняется корректно

# Сохраняем результат
out.save("ready.png")
