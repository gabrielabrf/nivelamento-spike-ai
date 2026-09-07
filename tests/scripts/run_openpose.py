import os
import time
import json
import cv2
import numpy as np
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

INPUT_DIR = os.path.join(TESTS_DIR, "input")
OUTPUT_DIR = os.path.join(TESTS_DIR, "output", "openpose")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aponta para a raiz do projeto onde os arquivos da sua imagem estão localizados
PROTO_FILE = os.path.join(PROJECT_ROOT, "openpose_coco.prototxt")
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, "pose_deploy_linevec.caffemodel")

POSE_PAIRS = [
    [1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
    [0,14],[0,15],[14,16],[15,17]
]

def load_openpose_net():
    if not os.path.exists(PROTO_FILE) or not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Arquivos não encontrados na raiz do projeto:\n"
            f"1. {PROTO_FILE}\n2. {WEIGHTS_FILE}"
        )
    
    net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)
    
    # Configuração direta para CPU (evita o erro do backend CUDA no OpenCV)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return net

def process_video(video_path, net):
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_pose.mp4")
    output_metrics_path = os.path.join(OUTPUT_DIR, f"{base_name}_metrics.json")
    
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))
    
    frame_count = 0
    inference_times = []
    cpu_usages = []
    
    in_width, in_height = 368, 368
    
    print(f"Processando OpenPose: {video_name}...")
    start_total_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        
        t0 = time.time()
        output = net.forward()
        t1 = time.time()
        
        inference_times.append(t1 - t0)
        cpu_usages.append(psutil.cpu_percent())
        
        H = output.shape[2]
        W = output.shape[3]
        
        points = []
        for i in range(18):
            probMap = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(probMap)
            x = (width * point[0]) / W
            y = (height * point[1]) / H
            points.append((int(x), int(y)) if prob > 0.1 else None)
                
        for pair in POSE_PAIRS:
            partA, partB = pair[0], pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 0, 0), 2)
                cv2.circle(frame, points[partA], 4, (0, 0, 255), -1)
                cv2.circle(frame, points[partB], 4, (0, 0, 255), -1)
                
        inst_fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        cv2.putText(frame, f"OpenPose FPS: {inst_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        out.write(frame)
        
    total_processing_time = time.time() - start_total_time
    cap.release()
    out.release()
    
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    metrics = {
        "modelo": "OpenPose (OpenCV Caffe)",
        "video": video_name,
        "frames_totais": frame_count,
        "tempo_total_segundos": round(total_processing_time, 2),
        "tempo_medio_inferencia_ms": round(avg_inference_time * 1000, 2),
        "fps_medio": round(avg_fps, 2),
        "cpu_medio_percentual": round(sum(cpu_usages) / len(cpu_usages), 2) if cpu_usages else 0,
        "gpu_utilizada": False
    }
    
    with open(output_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    print(f"Concluído: {video_name} | FPS Médio: {metrics['fps_medio']}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Diretório {INPUT_DIR} não encontrado.")
        return
        
    try:
        net = load_openpose_net()
    except Exception as e:
        print(f"Erro ao carregar OpenPose: {e}")
        return
        
    videos = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    for v in videos:
        process_video(os.path.join(INPUT_DIR, v), net)

if __name__ == "__main__":
    main()