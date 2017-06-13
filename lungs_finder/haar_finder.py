import cv2


def __find_max_rectangle(rectangles):
    max_rectangle = None
    max_area = 0

    for rectangle in rectangles:
        x, y, width, height = rectangle
        area = width * height

        if area > max_area:
            max_area = area
            max_rectangle = rectangle

    return max_rectangle


def find_left_lung(image):
    left_lung = cv2.CascadeClassifier("lungs_finder/left_lung.xml")
    found = left_lung.detectMultiScale(image, 1.2, 5)
    left_lung_rectangle = __find_max_rectangle(found)

    return left_lung_rectangle
