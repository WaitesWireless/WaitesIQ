import sys
import cv2
import time
from modeling.inference import process_image
from modeling.database import Database
import argparse

def detect_camera_sources(max_sources=10):
    """Detect available camera indices."""
    available = []
    for idx in range(max_sources):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available

def main():
    parser = argparse.ArgumentParser(description="WaitesIQ Camera Inference")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the database directory")
    parser.add_argument("--camera", type=int, default=1, help="Camera index")
    args = parser.parse_args()
    db = Database(args.db_path)

    camera_index = args.camera

    # Detect and print available cameras
    cameras = detect_camera_sources()
    if camera_index not in cameras:
        print(f"Warning: Camera index {camera_index} is not available.")
        camera_index = cameras[0] if cameras else None
    print(f"Available camera indices: {cameras}")

    # Check if camera opened successfully
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.camera_index}.")
        return

    print("Press 'q' to quit. Press 'space' to save frame as PNG.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        boxes, embeddings = process_image(frame)

        if boxes is not None:
            for box, embedding in zip(boxes, embeddings):

                area = (box[2]-box[0]) * (box[3]-box[1])
                # too far away, don't do anything with it
                if area < 1000:
                    continue

                # Draw bounding box on the frame
                cv2.rectangle(frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              (255, 0, 0), 2)
                
                name, iq = db.search_waites(embedding)
                if name is None:
                    # name, iq = db.search_star_wars(embedding)
                    name, iq = db.add_unknown(embedding)

                lines = [f"Name: {name.replace('_', ' ').title()}", f"IQ: {iq}"]
                text_x = int(box[0])
                text_y = int(box[3]) + 30  # Starting y position

                for i, line in enumerate(lines):
                    y = text_y + i * 30  # 30 pixels between lines
                    cv2.putText(frame, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space key
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"frame_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    # Release everything
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        sys.exit(0)