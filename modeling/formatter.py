import cv2

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, img, img_copy, rect_done
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img = img_copy.copy()
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        img = img_copy.copy()
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)
        rect_done = True

def crop_and_resize(img, x1, y1, x2, y2, size=(160, 160)):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    # Make square
    w = x_max - x_min
    h = y_max - y_min
    side = min(w, h)
    x_max = x_min + side
    y_max = y_min + side
    cropped = img[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, size)
    return resized

if __name__ == "__main__":
    image_path = "C:/Users/jake.kasper_waites/Work/Repos/WaitesIQ/griev.jpg"  # Change to your image path
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        exit(1)
    img_copy = img.copy()
    drawing = False
    rect_done = False
    ix = iy = fx = fy = -1

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        if rect_done:
            cropped_resized = crop_and_resize(img_copy, ix, iy, fx, fy)
            cv2.imshow("Cropped & Resized", cropped_resized)
            rect_done = False

            # Wait for key press on the cropped image
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('s'):
                save_path = "cropped_resized.jpg"
                cv2.imwrite(save_path, cropped_resized)
                print(f"Image saved to {save_path}")
                cv2.destroyWindow("Cropped & Resized")

    cv2.destroyAllWindows()