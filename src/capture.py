import cv2

def Capture() :
	
    # 1.creating a video object
    video = cv2.VideoCapture(0)
    check, frame = video.read()
	
    # 7. image saving
    showPic = cv2.imwrite("template.jpg",frame)
    
    # 8. shutdown the camera
    video.release()
	
    img_orig = cv2.imread('template.jpg')
    img_gray = cv2.imread('template.jpg',0)

    return img_orig, img_gray


