import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from operator import itemgetter

NEWH, NEWW = 320, 320
MIN_CONFIDENCE = 0.5

def box_in_box(b, outer_box) :
    x1, y1, x2, y2 = outer_box
    return (x1-5) < b[0] < (x2) and (x1) < b[0] + b[2] < (x2+5) and (y1-5) < b[1] < (y2) and (y1) < b[3]+ b[1] < (y2+5)     


def mser_regs(gray_img) :
    mser = cv2.MSER_create(8,80, 8000)
    msers, bboxes = mser.detectRegions(gray_img)
    return msers, bboxes


def east(image, H, W) :
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < MIN_CONFIDENCE:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    bboxes = non_max_suppression(np.array(rects), probs=confidences)
    return bboxes


def rescale(boxes, rW, rH) :
    return [[int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)] for (startX, startY, endX, endY) in boxes] 


def link_letters_to_words(mser_boxes, bboxes) :
    letters = [[] for i in range(len(bboxes))]
    for mser_box in mser_boxes :
        for i in range(len(bboxes)) :
            if box_in_box(mser_box, bboxes[i]) :
                letters[i].append(mser_box)
                break
    return letters

def word_seq(bboxes) :
    bboxes = sorted(bboxes, key = itemgetter(1))
    j, sorted_bboxes = 0, []
    for i in range(len(bboxes) - 1) :
        if abs(bboxes[i + 1][1] - bboxes[i][1]) > 5 :
            sorted_bboxes += sorted(bboxes[j:i+1], key = itemgetter(0))
            j = i + 1
    sorted_bboxes += sorted(bboxes[j:], key = itemgetter(0))
    return sorted_bboxes    

def get_bboxes(orig_img, gray_img):
    image = orig_img.copy()
    (H, W) = image.shape[:2]

    rW = W / float(NEWW)
    rH = H / float(NEWH)

    image = cv2.resize(image, (NEWW, NEWH))
    (H, W) = image.shape[:2]

    bboxes = east(image, H, W)
    msers = mser_regs(gray_img)
    mser_boxes = msers[1]

    ordered_words = word_seq(rescale(bboxes, rW, rH))
    letters = link_letters_to_words(mser_boxes, ordered_words)

    #for (startX, startY, endX, endY) in ordered_words:
       # cv2.rectangle(orig_img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    #cv2.imshow("Text Detection", orig_img)
    #cv2.waitKey(0)

    letter_imgs = [[orig_img[b[1] : b[1] + b[3] , b[0] : b[0] + b[2] ] for b in _] for _ in letters] 
    word_imgs = [orig_img[b[1] : b[3], b[0] : b[2]] for b in ordered_words]

    return letter_imgs, word_imgs


