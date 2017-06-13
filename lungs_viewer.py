import os
import sys
import numpy as np
import cv2
import lungs_finder as lf

try:
    __import__("imp").find_module("dicompylercore")
    from dicompylercore import dicomparser

    dicom_support = True
except ImportError:
    dicom_support = False
    print("\"dicompylercore\" library was not found. \"lungs-finder\" works without dicom support.")


def parse_argv(argv):
    if len(argv) < 4:
        print("Usage: lungs_viewer.py \"path_to_folder\" \"position_to_start\" \"histogram_equalization\".")
        exit(1)

    path_to_folder = argv[1]
    position_to_start = int(argv[2])

    if argv[3].lower() in ["true", "1"]:
        histogram_equalization = True
    else:
        histogram_equalization = False

    return path_to_folder, position_to_start, histogram_equalization


def proportional_resize(image, max_side):
    if image.shape[0] > max_side or image.shape[1] > max_side:
        if image.shape[0] > image.shape[1]:
            height = max_side
            width = int(height / image.shape[0] * image.shape[1])
        else:
            width = max_side
            height = int(width / image.shape[1] * image.shape[0])
    else:
        height = image.shape[0]
        width = image.shape[1]

    return cv2.resize(image, (width, height))


def scan(argv):
    path_to_folder, position_to_start, histogram_equalization = parse_argv(argv)
    left_detects = 0
    left_total = 0
    walks = list(os.walk(path_to_folder))
    i = 0
    k = 1

    while i < len(walks):
        path, directories, files = walks[i]
        files = [file for file in files if not file[0] == "."]
        j = 0

        while j < len(files):
            file = files[j]

            if k < position_to_start:
                k += 1
                j += 1
            else:
                _, extension = os.path.splitext(file)

                if extension == ".dcm" and dicom_support:
                    parsed = dicomparser.DicomParser(path + os.sep + file)
                    image = np.array(parsed.GetImage(), dtype=np.uint8)

                    if parsed.GetImageData()["photometricinterpretation"] == "MONOCHROME1":
                        image = 255 - image

                    if histogram_equalization:
                        image = cv2.equalizeHist(image)
                        image = cv2.medianBlur(image, 3)

                elif extension in [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", ".jpe", ".png",
                                   ".tiff", ".tif"]:
                    image = cv2.imread(path + os.sep + file, 0)

                    if histogram_equalization:
                        image = cv2.equalizeHist(image)
                        image = cv2.medianBlur(image, 3)
                else:
                    j += 1
                    continue

                scaled_image = proportional_resize(image, 512)
                left_lung_rectangle = lf.find_left_lung(scaled_image)
                color_image = scaled_image

                if left_lung_rectangle is not None:
                    x, y, width, height = left_lung_rectangle
                    color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(color_image, (x, y), (x + width, y + height), (128, 128, 0), 2)
                    left_detects += 1

                left_total += 1
                cv2.imshow("lungs-finder", color_image)
                code = cv2.waitKey(0)

                while code not in [2, 3, 27, 32]:
                    code = cv2.waitKey(0)

                if code == 27:
                    print("Left lung rate: " + str(left_detects / left_total * 100) + "%.")
                    exit(0)
                elif code in [3, 32]:
                    j += 1
                else:
                    if j > 0:
                        j -= 1
                    else:
                        if i > 0:
                            i -= 2
        i += 1

    print("Left lung rate: " + str(left_detects / left_total * 100) + "%.")


if __name__ == "__main__":
    sys.exit(scan(sys.argv))
