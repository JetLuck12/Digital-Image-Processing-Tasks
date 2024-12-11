import cv2

video_path = 'peshih.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка загрузки видео.")
    exit()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrians, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Pedestrian Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
