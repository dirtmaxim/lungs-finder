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
    if len(argv) != 3:
        print("Usage: lungs_viewer.py \"path_to_folder\" \"use_labels\".")
        exit(1)

    path_to_folder = argv[1]

    if argv[2].lower() in ["true", "1"]:
        use_labels = True
    else:
        use_labels = False

    return path_to_folder, use_labels


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
    path_to_folder, use_labels = parse_argv(argv)
    walks = list(os.walk(path_to_folder))
    i = 0

    while i < len(walks):
        path, directories, files = walks[i]
        files = [file for file in files if not file[0] == "."]
        j = 0

        while j < len(files):
            file = files[j]

            _, extension = os.path.splitext(file)

            if extension == ".dcm" and dicom_support:
                parsed = dicomparser.DicomParser(path + os.sep + file)
                image = np.array(parsed.GetImage(), dtype=np.uint8)

                if parsed.GetImageData()["photometricinterpretation"] == "MONOCHROME1":
                    image = 255 - image

                image = cv2.equalizeHist(image)
                image = cv2.medianBlur(image, 3)
            elif extension in [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", ".jpe", ".png",
                               ".tiff", ".tif"]:
                image = cv2.imread(path + os.sep + file, 0)
                image = cv2.equalizeHist(image)
                image = cv2.medianBlur(image, 3)
            else:
                j += 1
                continue

            scaled_image = proportional_resize(image, 512)
            right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
            left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
            right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
            left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
            right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
            left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
            color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

            if right_lung_hog_rectangle is not None:
                x, y, width, height = right_lung_hog_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
                if use_labels:
                    cv2.putText(color_image, "HOG Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (59, 254, 211), 1)

            if left_lung_hog_rectangle is not None:
                x, y, width, height = left_lung_hog_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
                if use_labels:
                    cv2.putText(color_image, "HOG Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (59, 254, 211), 1)

            if right_lung_lbp_rectangle is not None:
                x, y, width, height = right_lung_lbp_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
                if use_labels:
                    cv2.putText(color_image, "LBP Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (130, 199, 0), 1)

            if left_lung_lbp_rectangle is not None:
                x, y, width, height = left_lung_lbp_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
                if use_labels:
                    cv2.putText(color_image, "LBP Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (130, 199, 0), 1)

            if right_lung_haar_rectangle is not None:
                x, y, width, height = right_lung_haar_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
                if use_labels:
                    cv2.putText(color_image, "HAAR Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (245, 199, 75), 1)

            if left_lung_haar_rectangle is not None:
                x, y, width, height = left_lung_haar_rectangle
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
                if use_labels:
                    cv2.putText(color_image, "HAAR Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (245, 199, 75), 1)

            cv2.imshow("lungs-finder", color_image)
            found_lungs = lf.get_lungs(scaled_image, 10)

            if found_lungs is not None:
                cv2.imshow("Found lungs", found_lungs)

            code = cv2.waitKey(0)

            while code not in [2, 3, 27, 32]:
                code = cv2.waitKey(0)

            if code == 27:
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


if __name__ == "__main__":
    sys.exit(scan(sys.argv))
