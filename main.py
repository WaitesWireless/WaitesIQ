import cv2
import time
from modeling.inference import process_image
from modeling.database import Database

def main():
    db = Database("C:/Users/jake.kasper_waites/Work/Repos/WaitesIQ/data")

    # Check if camera opened successfully
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
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
                # Draw bounding box on the frame
                cv2.rectangle(frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              (255, 0, 0), 2)
                
                name, iq = db.search_waites(embedding)

                if name is None:
                    name = db.search_star_wars(embedding)
                    text = name
                else:
                    text = f"{name}: {iq}"

                text_x = int(box[0])
                text_y = int(box[3]) + 30  # 30 pixels below the bottom of the box
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    main()