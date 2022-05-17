import cv2
import cvzone

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
cam.set(10,70)
# detector
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))

# Load labels and classifier
Labels = []
File = 'coco.names'
with open(File,'rt') as f:
    Labels = f.read().rstrip('\n').split('\n')

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight, config)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
timer = 0
while timer<300:
    timer = timer + 1
    # read video feed
    success, frame = cam.read()

    # detect objects
    Ids, confids, boundBox = net.detect(frame, confThreshold=0.5)

    # fshape = frame.shape
    # fheight = fshape[0]
    # fwidth = fshape[1]
    # print(fheight, fwidth)

    # if there is an object
    if len(Ids) != 0:
        for id, confidence, bbox in zip(Ids.flatten(), confids.flatten(), boundBox):
            cv2.rectangle(frame, bbox, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, Labels[id-1].upper(), (bbox[0] + 10, bbox[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(confidence * 100, 2)), (bbox[0] + 200, bbox[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("video", frame)
    cv2.waitKey(1)
else:
    cam.release()
    out.release()