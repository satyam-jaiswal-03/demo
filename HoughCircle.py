import cv2
import numpy as np
import time

def is_ellipse(contour, min_aspect=0.7, max_aspect=0.95, min_area=300):
    if len(contour) < 5:
        return False, None
    ellipse = cv2.fitEllipse(contour)
    (_, _), (MA, ma), _ = ellipse
    aspect_ratio = min(MA, ma) / max(MA, ma)
    area = np.pi * (MA / 2) * (ma / 2)
    if min_aspect <= aspect_ratio <= max_aspect and area > min_area:
        return True, ellipse
    return False, None

def is_good_circle(cnt, min_circularity=0.8, min_area=300):
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity >= min_circularity

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # --- Circle Detection (HoughCircles) ---
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=40, minRadius=10, maxRadius=80
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, "Circle", (x - r, y - r - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Preprocessing for contour-based detection ---
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Morphological clean-up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 5:
            continue

        if is_good_circle(cnt):
            (x, y), r = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(r)
            cv2.circle(output, center, radius, (0, 255, 255), 2)
            cv2.putText(output, "Circle", (center[0] - radius, center[1] - radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            continue

        is_ell, ellipse = is_ellipse(cnt)
        if is_ell:
            cv2.ellipse(output, ellipse, (255, 0, 0), 2)
            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.putText(output, "Ellipse", (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Well-defined Circle & Ellipse Detection', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
