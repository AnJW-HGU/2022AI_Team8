import cv2
import numpy as np
from dynamikontrol import Module

# 90% 이상일 경우, 차 또는 번호판으로 인식 (0~1)
CONFIDENCE = 0.9 
# Non-Maximum Suppression의 Threshold (YOLO 결과값 압축 때 사용)
THRESHOLD = 0.3
# 라벨
LABELS = ['Car', 'Plate']
# 자동차의 크기가 500일 때 모터 제어
CAR_WIDTH_TRESHOLD = 500

# 영상일 경우 영상 path로
cap = cv2.VideoCapture(0)

# 다크넷의 모델을 읽어오는 (모델 설정값, 모델 weight)
net = cv2.dnn.readNetFromDarknet('cfg/yolov4-ANPR.cfg', 'yolov4-ANPR.weights')

# dinamikontrol inital
module = Module()

# 웹캠 열기
while cap.isOpened():
    # 이미지 읽기
    ret, img = cap.read()
    if not ret:
        break
    
    # 이미지의 세로, 가로 사이즈
    H, W, _ = img.shape
    
    
    # (Darknet 설정 또는 컴파일 어려울 경우)
    #
    # 전처리 (OpenCV = BGR, Darknet = RGB)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)
    # 전처리 이미지를 입력값으로
    net.setInput(blob)
    # Output
    output = net.forward()
    
    # Bounding Box(네모칸 저장), CONFIDENCES, Class ID 
    boxes, confidences, class_ids = [], [], []
    
    # 차가 여러개의 경우 output이 list의 형태로 나오게 됨
    for det in output:
        # det : [x, y, w, h, scores~~]
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            # box의 값이 0~1로 normalize 되어있음
            # 픽셀의 너비를 곱해줘야함
            cx, cy, w, h = box * np.array([W, H, W, H])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Non-Maximum Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            
            # Drawing
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            
            # 모터 부분
            # class_ids == 0 (car), class_ids == 1 (plate)
            if class_ids[i] == 0:
                if w > CAR_WIDTH_TRESHOLD:
                    # module.motor.angle(80)
                    # 차라고 인식 했을 경우 : 
                    # 번호판의 글자를 인식하는 걸 만들어야함
                    
                # else:
                    # module.motor.angle(0)
    # else:
        # module.motor.angle(0)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
