import cv2
import numpy as np
import math

def order_coordinates(coords):
    sortX = coords[np.argsort(coords[:,0]),:]
    left = sortX[:2]
    right = sortX[2:]

    left = left[np.argsort(left[:,1]),:]
    topLeft = left[0,:]
    bottomLeft = left[1,:]

    right = right[np.argsort(right[:,1]),:]
    topRight = right[0,:]
    bottomRight = right[1,:]

    return np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")

def transformRectangle(image, coords, flip=False):
    ordered = order_coordinates(coords)
    topLeft = ordered[0,:]
    topRight = ordered[1,:]
    bottomRight = ordered[2,:]
    bottomLeft = ordered[3,:]

    bottomWidth = np.sqrt(math.pow(bottomRight.item(0) - bottomLeft.item(0), 2) + math.pow(bottomRight.item(1) - bottomLeft.item(1), 2))
    topWidth = np.sqrt(math.pow(topRight.item(0) - topLeft.item(0), 2) + math.pow(topRight.item(1) - topLeft.item(1), 2))
    width = max(int(topWidth), int(bottomWidth))
    leftHeight = math.sqrt(math.pow(topLeft.item(0) - bottomLeft.item(0), 2) + math.pow(topLeft.item(1) - bottomLeft.item(1), 2))
    rightHeight = math.sqrt(math.pow(topRight.item(0) - bottomRight.item(0), 2) + math.pow(topRight.item(1) - bottomRight.item(1), 2))
    height = max(int(leftHeight), int(rightHeight))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(ordered, dst)
    transformed = cv2.warpPerspective(image, m, (width, height))
    if (flip):
        transformed = cv2.flip(transformed, 0)
    return transformed
