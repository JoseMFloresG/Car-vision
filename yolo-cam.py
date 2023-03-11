from ultralytics import YOLO 
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0) #Iniciamos la webcam
#Aqui ingresamos la resolucion de la webcam
cap.set(3, 1280) #3 indica ancho
cap.set(4,720) #4 indica alto

#cap = cv2.VideoCapture("sVideos/bikes.mp4") #For Video

model = YOLO("weights/yolov8n.pt") #Mandamos a llamar a los pesos predefinidos por YOLO

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

zone = "middle"

while True:
    succes, img = cap.read() #La funcion read regresa a succes si se recibio una imagen y a img la imagen recibida
    results = model(img, stream=True) #Comparamos las imagenes recibidas con los pesos
    
    detections = np.empty((0, 5))

    #El siguiente for dependiendo de las imagenes obtenidas muestra los rectangulos alrededor de los objetos escaneado
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Con opencv
            x1,y1,x2,y2 = box.xyxy[0] #Guardamos los valores de alto, ancho, x y y
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2) #Convertimos los valores a ints
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),4) #Creamos los rectangulos de los objetos

            # Con cvzone
            w,h = x2-x1,y2-y1 #Guardamos los valores de alto, ancho, x y y
            #bbox = int(x1), int(y1), int(w), int(h) #Convertimos los valores a ints
            if classNames[0]:
                print("x1= "+str(x1), "y1="+str(y1), "w="+str(w), "h="+str(h))
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100 #Obtenemos el weight del objeto y lo redondeamos
            cls = int(box.cls[0])

            # Only detect persons
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.5:
                cvzone.cornerRect(img,(x1,y1,w,h), l=10, rt=5)#Creamos los rectangulos de los objetos
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=2,thickness=2) #Imprimimos el weight del objeto en una etiqueda encima del rectangulo del objeto

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    
            # Calculate the central pixel
            sup_izq,sup_der,inf_izq,inf_der = (x1),(x1+w),(x1),(x1+w)
            prom_pixel = (sup_izq+sup_der+inf_izq+inf_der)/4
             
            #  Generate the divisions on the screen
            if prom_pixel >= 0 and prom_pixel <= 426:
                zone = "left"
            elif prom_pixel > 426 and prom_pixel < 854:
                zone = "midle"
            else:
                zone = "right"
            print(zone)
            

            
                



    cv2.imshow("Image", img) #Mostramos lo que ve la webcam
    cv2.waitKey(1)

