import os
import time
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import psutil
import os
import tensorflow as tf

# Silencia avisos do C++ do TensorFlow e avisos de funções depreciadas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Diretórios ajustados para a estrutura tests/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(TESTS_DIR, "input")
OUTPUT_DIR = os.path.join(TESTS_DIR, "output", "movenet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Carregando modelo MoveNet Thunder...")
module = tfhub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = module.signatures['serving_default']

KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    height, width, _ = frame.shape
    for edge in KEYPOINT_EDGES:
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
    for kp in keypoints:
        y, x, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x * width), int(y * height)), 5, (0, 0, 255), -1)

def process_video(video_path):
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
    
    print(f"Processando MoveNet: {video_name}...")
    start_total_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(tf.expand_dims(img_rgb, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        t0 = time.time()
        outputs = movenet(input_image)
        t1 = time.time()
        
        inference_times.append(t1 - t0)
        cpu_usages.append(psutil.cpu_percent())
        
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        draw_skeleton(frame, keypoints)
        
        inst_fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        cv2.putText(frame, f"MoveNet FPS: {inst_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(frame)
        
    total_processing_time = time.time() - start_total_time
    cap.release()
    out.release()
    
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    metrics = {
        "modelo": "MoveNet Thunder",
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
        
    videos = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    for v in videos:
        process_video(os.path.join(INPUT_DIR, v))

if __name__ == "__main__":
    main()