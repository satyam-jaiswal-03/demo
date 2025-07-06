import cv2
import numpy as np
import time

# Parameters
CIRCLE_DEBOUNCE_FRAMES = 10
ELLIPSE_DEBOUNCE_FRAMES = 10
SMOOTHING_ALPHA = 0.2
MAX_FPS = 15

# Tracking memory
circle_memory = []  # Each item: [x, y, r, frames_left]
ellipse_memory = []  # Each item: [ellipse_params, frames_left]

# Helpers
def smooth(old, new):
    return SMOOTHING_ALPHA * new + (1 - SMOOTHING_ALPHA) * old

def is_good_circle(cnt, min_circularity=0.75, min_area=400):
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity >= min_circularity

def is_ellipse(cnt, min_aspect=0.6, max_aspect=0.98, min_area=400):
    if len(cnt) < 5:
        return False, None
    ellipse = cv2.fitEllipse(cnt)
    (x, y), (MA, ma), angle = ellipse
    aspect_ratio = min(MA, ma) / max(MA, ma)
    area = np.pi * (MA / 2) * (ma / 2)
    if min_aspect <= aspect_ratio <= max_aspect and area > min_area:
        return True, ellipse
    return False, None

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # --- HoughCircles ---
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=40, minRadius=15, maxRadius=100
    )
    new_circles = []
    if circles is not None:
        for x, y, r in np.round(circles[0, :]).astype(int):
            new_circles.append([x, y, r, CIRCLE_DEBOUNCE_FRAMES])

    # --- Contours ---
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_ellipses = []
    for cnt in contours:
        if is_good_circle(cnt):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            new_circles.append([int(x), int(y), int(radius), CIRCLE_DEBOUNCE_FRAMES])
        is_ell, ellipse = is_ellipse(cnt)
        if is_ell:
            new_ellipses.append([ellipse, ELLIPSE_DEBOUNCE_FRAMES])

    # --- Update memory ---
    updated_circles = []
    for x, y, r, _ in new_circles:
        matched = False
        for idx, (px, py, pr, life) in enumerate(circle_memory):
            if abs(px - x) < 40 and abs(py - y) < 40:
                smoothed_x = int(smooth(px, x))
                smoothed_y = int(smooth(py, y))
                smoothed_r = int(smooth(pr, r))
                updated_circles.append([smoothed_x, smoothed_y, smoothed_r, CIRCLE_DEBOUNCE_FRAMES])
                circle_memory[idx][-1] = 0  # Reset aging
                matched = True
                break
        if not matched:
            updated_circles.append([x, y, r, CIRCLE_DEBOUNCE_FRAMES])
    circle_memory = updated_circles

    for c in circle_memory:
        c[3] -= 1
    circle_memory = [c for c in circle_memory if c[3] > 0]

    for x, y, r, _ in circle_memory:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.putText(output, "Circle", (x - 20, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    updated_ellipses = []
    for e, _ in new_ellipses:
        matched = False
        for idx, (pe, life) in enumerate(ellipse_memory):
            (x1, y1), _, _ = e
            (x2, y2), _, _ = pe
            if abs(x1 - x2) < 40 and abs(y1 - y2) < 40:
                updated_ellipses.append([e, ELLIPSE_DEBOUNCE_FRAMES])
                ellipse_memory[idx][-1] = 0
                matched = True
                break
        if not matched:
            updated_ellipses.append([e, ELLIPSE_DEBOUNCE_FRAMES])
    ellipse_memory = updated_ellipses

    for e in ellipse_memory:
        e[1] -= 1
    ellipse_memory = [e for e in ellipse_memory if e[1] > 0]

    for ellipse, _ in ellipse_memory:
        cv2.ellipse(output, ellipse, (255, 0, 0), 2)
        center = tuple(map(int, ellipse[0]))
        cv2.putText(output, "Ellipse", (center[0] - 30, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Stable Circle & Ellipse Detection", output)

    elapsed = time.time() - loop_start
    sleep_time = max(0, (1.0 / MAX_FPS) - elapsed)
    time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()