import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
prev_time = time.time()
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny for better contour edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # Skip very small or degenerate contours
        if len(cnt) < 5:
            continue

        # Fit ellipse
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse  # major/minor axes
        aspect_ratio = ma / MA if MA != 0 else 0

        # Solidity check to ensure shape isn't too ragged
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area

        # Filter based on aspect ratio and solidity
        if 0.7 <= aspect_ratio <= 1.3 and solidity > 0.85:
            # Draw ellipse
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        elif 0.4 <= aspect_ratio < 0.7 and solidity > 0.85:
            # Elliptical but not circle-like
            cv2.ellipse(frame, ellipse, (0, 200, 255), 2)

    # FPS calculation (smoothed)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    fps_list.append(fps)
    if len(fps_list) > 10:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Circular & Elliptical Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
