import cv2
import os

vidPath1 = "data\\1728090.mp4"
vidcap1 = cv2.VideoCapture(vidPath1)
success1,image1 = vidcap1.read()
count = 0
path="data\\1728090"

while success1:
    cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), image1)
    count += 1
    success1,image1 = vidcap1.read()
    print('Read a new frame: ', success1)

vidcap1.release()
cv2.destroyAllWindows()