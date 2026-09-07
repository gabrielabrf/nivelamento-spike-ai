import os
import time
import json
import cv2
import numpy as np
import torch
import torchvision

# Caminhos dinâmicos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_DIR = os.path.join(TESTS_DIR, "input")
OUTPUT_DIR = os.path.join(TESTS_DIR, "output", "mmpose")
MODELS_DIR = os.path.join(TESTS_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Esqueleto COCO (17 keypoints)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]

def load_keypoint_model():
    """Carrega um modelo leve de detecção de keypoints nativo do torchvision para evitar erros de rede/MMCV."""
    print("Carregando modelo de Keypoints via PyTorch (Keypoint R-CNN)...")
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

def run_mmpose_pipeline(video_path, model):
    video_name = os.path.basename(video_path)
    print(f"Processando MMPose/Keypoint-Pipeline: {video_name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, f"mmpose_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    start_time = time.time()
    processed_frames = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1

        # Conversão para Tensor do PyTorch
        img_tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            outputs = model([img_tensor])

        if len(outputs) > 0 and len(outputs[0]['keypoints']) > 0:
            # Pega a detecção com maior pontuação
            scores = outputs[0]['scores'].cpu().numpy()
            if len(scores) > 0 and scores[0] > 0.3:
                detected_frames += 1
                keypoints = outputs[0]['keypoints'][0].cpu().numpy() # (17, 3) -> [x, y, visibility]

                # Desenha pontos
                for kp in keypoints:
                    x, y, v = int(kp[0]), int(kp[1]), kp[2]
                    if v > 0.3:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Desenha conexões do esqueleto
                for p1, p2 in SKELETON:
                    if keypoints[p1][2] > 0.3 and keypoints[p2][2] > 0.3:
                        pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                        pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        out_video.write(frame)

    cap.release()
    out_video.release()

    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0
    detection_rate = (detected_frames / processed_frames * 100) if processed_frames > 0 else 0

    metrics = {
        "video": video_name,
        "total_frames": processed_frames,
        "total_time_seconds": round(total_time, 4),
        "avg_fps": round(avg_fps, 2),
        "detection_rate_percent": round(detection_rate, 2)
    }

    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{os.path.splitext(video_name)[0]}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Concluído em {round(total_time, 2)}s | FPS Médio: {round(avg_fps, 2)} | Detecção: {round(detection_rate, 2)}%\n")

def main():
    model = load_keypoint_model()

    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not videos:
        print(f"Nenhum vídeo encontrado em: {INPUT_DIR}")
        return

    print(f"Encontrados {len(videos)} vídeos na pasta input.\n")
    for video in videos:
        run_mmpose_pipeline(os.path.join(INPUT_DIR, video), model)

if __name__ == "__main__":
    main()