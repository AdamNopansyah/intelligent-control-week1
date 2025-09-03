import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Unpack the return value
    if not ret:
        print("Error: Could not read frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna merah dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Rentang warna biru dalam HSV
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])

    # Rentang warna hijau dalam HSV
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])

    # Rentang warna kuning dalam HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Masking untuk mendeteksi warna merah, biru, hijau, dan kuning
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = frame.copy()

    # Menemukan kontur untuk warna merah
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10000:  # Hanya tampilkan bounding box yang besar
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Warna merah
            cv2.putText(result, "Merah", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Menemukan kontur untuk warna biru
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10000:  # Hanya tampilkan bounding box yang besar
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Warna biru
            cv2.putText(result, "Biru", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Menemukan kontur untuk warna hijau
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10000:  # Hanya tampilkan bounding box yang besar
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Warna hijau
            cv2.putText(result, "Hijau", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Menemukan kontur untuk warna kuning
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10000:  # Hanya tampilkan bounding box yang besar
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Warna kuning
            cv2.putText(result, "Kuning", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Menampilkan hasil
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
