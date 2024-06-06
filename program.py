import cv2
from ultralytics import YOLO

model = YOLO('yolo_model_trained/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detecting using YOLOv8 image
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas do retângulo
            conf = box.conf[0]  # Confiança da detecção
            cls = box.cls[0]  # Classe do objeto

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', frame)

    # Exit when 'q' where pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
