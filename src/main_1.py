import capture
import bounding_boxes
#import ocr
import tts
import pytesseract
import cv2

print("Enter 1 for book reading \n Enter 2 for scene images \n -->")
a = int(input())

#orig_img, gray_img = capture.Capture()
img23 = cv2.imread('DNO.jpg')
gray23 = cv2.imread('DNO.jpg', 0)

orig_img, gray_img = img23, gray23

custom_config = r'--oem 3 --psm 6'

if a == 1 :
    text = pytesseract.image_to_string(orig_img, config = custom_config)

elif a == 2 :
    letters, words = bounding_boxes.get_bboxes(orig_img, gray_img)
    text = ' '.join([pytesseract.image_to_string(word, config = custom_config) for word in words if word.shape[1] > 0 and word.shape[1] > 0])

print(text)
#text = ocr.OCR(letters)
tts.texttospeech(text)
