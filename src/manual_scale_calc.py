import cv2
from functionality import load_images

from config import parameters


def mouse_hover(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_MOUSEMOVE:
        if len(img.shape) == 3:
            pixel_val = img[y, x]
            print(f"Pixel Coordinates: (x={x}, y={y}), Pixel Value (BGR): {pixel_val}")
        else:
            pixel_val = img[y, x]
            print(f"Pixel Coordinates: (x={x}, y={y}), Pixel Value: {pixel_val}")


folder_path = 'C:/Users/Aurora/OneDrive - KU Leuven/Modules/4th Semester/images/kuleuven_sem/Konstantinos/extracted raw images/Konstantinos'
images_and_names = load_images(folder_path)

if images_and_names:
    for img, name in images_and_names:
        # Resize the image to 1600x1200 before displaying
        resized_img = cv2.resize(img, (1400, 1000))

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a window that can be resized
        cv2.setMouseCallback("Image", mouse_hover, param=resized_img)

        if parameters["show_images"]:
            cv2.imshow("Image", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
else:
    print("No images loaded.")
