import cv2
import numpy as np
import mediapipe as mp
from src.pose import pose
from src.video import obter_lista_de_videos, abrir_video
from src.yolo import yolo, yolodetec

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def desenhar_pose(image, pose_landmarks):
    h, w, _ = image.shape
    coords = []
    
    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            cv2.line(image, pt1, pt2, (0, 255, 255), 3)

def detectar_pose_do_mais_proximo(detector, frame, possivel_proximo, padding_ratio=0.15):
    if possivel_proximo is None:
        return None

    x1, y1, x2, y2 = possivel_proximo
    frame_h, frame_w = frame.shape[:2]

    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(frame_w, x2 + pad_w)
    crop_y2 = min(frame_h, y2 + pad_h)

    recorte = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if recorte.size == 0:
        return None

    image_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    if not detection_result.pose_landmarks:
        return None

    landmarks = detection_result.pose_landmarks[0]
    recorte_h, recorte_w = recorte.shape[:2]

    landmarks_ajustados = []
    for lm in landmarks:
        x_abs = lm.x * recorte_w + crop_x1
        y_abs = lm.y * recorte_h + crop_y1
        
        class Point:
            pass
        p = Point()
        p.x = x_abs / frame_w
        p.y = y_abs / frame_h
        landmarks_ajustados.append(p)

    return landmarks_ajustados


def render():
    detector = pose()
    yolo_model, PERSON_CLASS, BALL_CLASS = yolo()
    
    lista_videos = obter_lista_de_videos()
    if not lista_videos:
        print("Nenhum vídeo encontrado na pasta 'input'.")
        return

    for nome_video in lista_videos:
        print(f"Processando vídeo: {nome_video}...")
        cap, out = abrir_video(nome_video)
        if cap is None:
            continue

        ultimo_jogador = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resultado = yolodetec(yolo_model, frame, PERSON_CLASS, BALL_CLASS)
            pessoas = []
            bola = None
            possivel_proximo = None
            menor_distancia = None

            for box in resultado.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == PERSON_CLASS:
                    pessoas.append((x1, y1, x2, y2))
                elif cls_id == BALL_CLASS:
                    bola = (x1, y1, x2, y2)

            if bola is not None:
                bola_cx = (bola[0] + bola[2]) / 2
                bola_cy = (bola[1] + bola[3]) / 2

                for p in pessoas:
                    pessoa_cx = (p[0] + p[2]) / 2
                    pessoa_cy = (p[1] + p[3]) / 2

                    distancia = ((pessoa_cx - bola_cx) ** 2 + (pessoa_cy - bola_cy) ** 2) ** 0.5
                    if possivel_proximo is None or distancia < menor_distancia:
                        possivel_proximo = p
                        menor_distancia = distancia

                if possivel_proximo is not None:
                    ultimo_jogador = possivel_proximo
            else:
                possivel_proximo = ultimo_jogador if ultimo_jogador is not None else (pessoas[0] if pessoas else None)

            frame_final = frame.copy()

            if possivel_proximo is not None:
                px1, py1, px2, py2 = possivel_proximo
                cv2.rectangle(frame_final, (px1, py1), (px2, py2), (255, 0, 0), 2)

                pose_landmarks = detectar_pose_do_mais_proximo(detector, frame, possivel_proximo)
                if pose_landmarks:
                    desenhar_pose(frame_final, pose_landmarks)

            out.write(frame_final)

        cap.release()
        out.release()
        print(f"Vídeo {nome_video} concluído!")

    detector.close()