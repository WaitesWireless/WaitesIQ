import cv2
import datetime

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        cv2.imshow('Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar pressed
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")

        elif key == ord('q'):  # 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()