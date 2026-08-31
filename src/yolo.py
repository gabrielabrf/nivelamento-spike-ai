from ultralytics import YOLO

def yolo():
    yolo_model = YOLO('yolov8n.pt')
    PERSON_CLASS = 0
    BALL_CLASS = 32
    return yolo_model, PERSON_CLASS, BALL_CLASS

def yolodetec(yolo_model, frame, PERSON_CLASS, BALL_CLASS):
    return yolo_model(frame, classes=[PERSON_CLASS, BALL_CLASS], verbose=False)[0]