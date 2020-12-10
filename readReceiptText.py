import pytesseract
from PIL import Image
import cv2
import re
from correctImage import correct

def readText(imagePath, flip=False):
    im = correct(imagePath, flip)
    cv2.imwrite("temp.png", im)
    #More pre processing?
    #Default oem/psm = 3
    text = pytesseract.image_to_string(im, lang="eng", config='--psm 3 --oem 3')
    lines = text.split('\n')
    name = lines[0]
    #Filter with regex, deal with different receipt configurations, possible NLP application
    # Current default: "itemName value" format
    # value format: optional $, digits . digits
    lineObjects = []
    for line in lines:
        match = re.search('\\b\$?\d+\.\d+\\b', line)
        if match:
            val = line[match.start():match.end()]
            item = line[:match.start()] + line[match.end():]
            lineObjects.append((item, val))

    print("Original:\n" + text)
    print("\nPlace: " + name)
    print("\nItems/Price:\n")
    print(lineObjects)
    return lineObjects

readText("test.png")
