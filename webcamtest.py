import cv2

# Change this number to test different webcams:
# 0 = usually built-in camera
# 1 = often USB webcam
# 2, 3, ... = other cameras / virtual cameras
CAMERA_INDEX = 1

# On Windows, CAP_DSHOW usually opens USB webcams more reliably
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Optional camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"Could not open camera index {CAMERA_INDEX}")
    exit()

print(f"Showing camera index {CAMERA_INDEX}. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break

    # Display index on the video feed
    cv2.putText(
        frame,
        f"Camera Index: {CAMERA_INDEX}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Webcam Test", frame)

    # Press q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()