import cv2
import numpy as np
from scipy.spatial import distance as dist

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Load the YOLO model and names file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Input video
cap = cv2.VideoCapture("s2.mp4")


# Social distance threshold (in pixels)
threshold = 100

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    i= 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite("face_" + str(i) + ".jpg", face_img)
        i += 1

    # Create a 4D blob from the video
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # Set the input to the YOLO model
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


    # Get the output from the YOLO model
    outs = net.forward(getOutputsNames(net))

    bboxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bboxes.append([x, y, w, h])
                confidences.append(confidence)


    # Parse the output and get the bounding boxes
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h, class_id])

    centroids = []
    for (x, y, w, h) in bboxes:
        centroid_x = x + w/2
        centroid_y = y + h/2
        centroids.append((centroid_x, centroid_y))


    

    # Perform non-maxima suppression to eliminate overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

    for (i, centroid1) in enumerate(centroids):
        for (j, centroid2) in enumerate(centroids):
            if i != j:
                distance = dist.euclidean(centroid1, centroid2)
                if distance < threshold:
                    radius = 5
                    print("Social distancing violation between faces ", i, " and ", j)
                    cv2.line(img, tuple(np.int0(centroid1)), tuple(np.int0(centroid2)), (0, 0, 255), 2)


                    cv2.circle(img, tuple(np.int0(centroid1)), radius, (0, 0, 255), -1)
                    # cv2.circle(img, tuple(centroid2), 5, (0, 0, 255), -1)

    # Draw the bounding boxes on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h, class_id = boxes[i]
            label = str(classes[class_id])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if label == "person":  
                violation_counter = 0              
                for (x2, y2, w2, h2, class_id2) in boxes:
                    
                    if class_id2 == class_id and (x, y, w, h) != (x2, y2, w2, h2):
                        
                        center_x1 = x + (w/2)
                        center_y1 = y + (h/2)
                        center_x2 = x2 + (w2/2)
                        center_y2 = y2 + (h2/2)
                        distance = np.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                        violation_counter += 1
    cv2.putText(img, "Social Distance Violations: " + str(violation_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

           
