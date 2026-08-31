import os
import cv2

def obter_lista_de_videos():
    input_folder = 'input'
    os.makedirs('output', exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)

    # Lista todos os arquivos .mp4 da pasta input
    arquivos = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    return arquivos

def abrir_video(nome_arquivo):
    input_video_path = os.path.join('input', nome_arquivo)
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
    output_video_path = os.path.join('output', f'{nome_sem_extensao}_pose.mp4')

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_video_path}")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    return cap, out