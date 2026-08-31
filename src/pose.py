import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def pose():
    model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task')
    
    # Baixa o modelo automaticamente se ele ainda não estiver na pasta
    if not os.path.exists(model_path):
        print("Baixando o modelo pose_landmarker_heavy.task...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download concluído!")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector