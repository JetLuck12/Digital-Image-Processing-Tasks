import numpy as np
from PIL import Image

noise = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
bg = Image.fromarray(noise, mode="L")
bg.save("bg.png")