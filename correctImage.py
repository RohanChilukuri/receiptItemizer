import cv2
import numpy as np
from rectTransform import transformRectangle

def correct(imagePath, flip=False):
    im = cv2.imread(imagePath)
    origHeight, origWidth = im.shape[:2]
    ratio = 500.0 / origHeight
    resizeHeight, resizeWidth = 500, int(ratio * origWidth)
    imResized = cv2.resize(im, (resizeWidth, resizeHeight))
    imGray = cv2.cvtColor(imResized, cv2.COLOR_BGR2GRAY)
    #Blurring/Filtering
    #imGray = cv2.GaussianBlur(imGray, (3, 3), 0) #Blurring #Adjust kernal size (3-7)
    imGray = cv2.medianBlur(imGray, 3)
    #imGray = cv2.bilateralFilter(imGray, 9, 75, 75)

    #Otsu method
    high_thresh = cv2.threshold(imGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    low_thresh = 0.5*high_thresh
    #Median method
    med = np.median(imGray)
    lowerThresh = int(max(0, .67 * med))
    higherThresh = int(min(255, 1.33 * med))
    #Canny Edge Detection with Automatic Canny thresholds
    imEdge = cv2.Canny(imGray, lowerThresh, higherThresh, L2gradient=True)

    #Dilations to close edge gaps
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #imEdge = cv2.dilate(imEdge, kernel)

    #Contours
    contours = cv2.findContours(imEdge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:10] #Largest contours
    found = np.array([(0, 0), (resizeWidth, 0), (resizeWidth, resizeHeight), (0, resizeHeight)])
    for contour in contours:
        epsilon = 0.02*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        if len(approx) == 4:
            found = approx
            break
    corrected = transformRectangle(im, found.reshape(4, 2) / ratio, flip)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    #corrected = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    corrected = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    corrected = cv2.medianBlur(corrected, 5)

    cv2.imshow("Original", im)
    cv2.imshow("Edge Detect", imEdge)
    cv2.imshow("Corrected", corrected)
    cv2.waitKey(0)

    return corrected
