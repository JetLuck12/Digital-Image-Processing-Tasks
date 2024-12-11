import cv2
import numpy as np

def filter_objects(objects):
    i = 0
    while i < len(objects):
        obj = objects[i]
        should_remove = False
        for j in range(len(objects)):
            if i != j and abs(obj['x'] - objects[j]['x']) <= 12:
                should_remove = True
                break

        if should_remove:
            objects.pop(i)
        else:
            i += 1

def nothing(x):
    pass

image_path = 'dota/sqr14.jpg'
img_rgb = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

templates = {}
for digit in range(10):
    template_path = f'dota/{digit}.png'
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    templates[digit] = template

window_name = 'Digit Detection'
cv2.namedWindow(window_name)
cv2.createTrackbar('Threshold', window_name, 87, 100, nothing)
exit_key = 'q'

while True:
    threshold = cv2.getTrackbarPos('Threshold', window_name) / 100.0
    img_display = img_rgb.copy()
    detections = []

    for digit, template in templates.items():
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            x, y = pt

            detections.append({'x': x, 'y': y, 'w': w, 'h': h, 'digit': digit})

    filter_objects(detections)
    detections = sorted(detections, key=lambda k: k['x'])
    digitals = []
    for det in detections:
        x, y, w, h, digit = det['x'], det['y'], det['w'], det['h'], det['digit']
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_display, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        digitals.append(det['digit'])
    if len(digitals) == 5:
        first_five = detections[:5]
        digits = [str(det['digit']) for det in first_five]
        time_str = f"{digits[0]}:{digits[1]}{digits[2]}:{digits[3]}{digits[4]}"
        cv2.putText(img_display, f"Time: {time_str}", (10, img_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
    elif len(detections) >= 4:
        first_four = detections[:4]
        digits = [str(det['digit']) for det in first_four]
        time_str = f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}"
        cv2.putText(img_display, f"Time: {time_str}", (10, img_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow(window_name, img_display)
    if cv2.waitKey(1) == ord(exit_key):
        break

cv2.destroyAllWindows()
