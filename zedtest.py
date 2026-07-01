import cv2

ZED_INDEX = 2   # change this after running zed_test.py
USE_LEFT = True # False = use right camera

cap = cv2.VideoCapture(ZED_INDEX, cv2.CAP_DSHOW)

# Try common ZED side-by-side resolution.
# If this fails, OpenCV will fall back to another supported mode.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {ZED_INDEX}")

print("Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    h, w = frame.shape[:2]

    # ZED stereo frame = left image + right image side-by-side.
    half_w = w // 2

    if USE_LEFT:
        mono = frame[:, :half_w]
    else:
        mono = frame[:, half_w:]

    cv2.imshow("ZED one camera only", mono)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()