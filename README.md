# lungs-finder

##### Library for lungs detection on CXR images

![](https://user-images.githubusercontent.com/11778655/27996147-765dfba8-64e4-11e7-81ca-9ea25b5d072d.png)

```
Requirements:
- Python 3;
- OpenCV3.
```

```
Installation:
- pip install lungs-finder
```

```
You can test your dataset using lungs_viewer.py.
It is a program that is displayed on the screenshot above.
Usage: lungs_viewer.py /your/dataset True
The last parameter defines whether labels will be displayed or not.
```

```
You can get lungs using HOG, Haar or LBP.
Example:

import lungs_finder as lf

image = cv2.imread("/your/image/cxr.png", 0)

# Get both lungs image. It uses HOG as main method,
# but if HOG found nothing it uses HAAR or LBP.
found_lungs = lf.get_lungs(image)

if found_lungs is not None:
    cv2.imshow("Found lungs", found_lungs)
    code = cv2.waitKey(0)

# Or you can get left or right lung independently using HOG, Haar or LBP.
right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)

if right_lung_hog_rectangle is not None:
    x, y, width, height = right_lung_hog_rectangle
    right_image = image[y:y + height, x:x + width]
    cv2.imshow("Right lung", right_image)
    code = cv2.waitKey(0)
```
