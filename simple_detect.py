import cv2
import numpy as np

# Load the pre-trained model and image
#model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')

# Pre-defined class names based on the Caffe model
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

camera = cv2.VideoCapture(-1) 
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
while True:
    # frame = picam2.capture_array() # Capture a frame from the camera
    _, frame = camera.read()
    # frame = cv2.flip(frame, 1) # if your camera reverses your image
    
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert the image from RGB to BGR because OpenCV uses BGR by default
    (h, w) = img.shape[:2]  # Get the height and width of the image
    # Generate the input blob for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)  # Set the blob as the input to the network
    detections = net.forward()  # Perform forward pass to get the detection results
    
    # Loop over the detected objects
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get the confidence of the detected object
        if confidence > 0.2:  # If the confidence is above the threshold, process the detected object
            idx = int(detections[0, 0, i, 1])  # Get the class index
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Get the bounding box of the object
            (startX, startY, endX, endY) = box.astype("int")  # Convert the bounding box to integers
    
            # Annotate the object and confidence on the image
            label = "{}: {:.2f}%".format(class_names[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
    cv2.imshow("Output", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
