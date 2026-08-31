# SpikeAI

## Descrição do projeto

O **SpikeAI** é um projeto de visão computacional voltado para a análise de movimento no esporte (com foco em vôlei). Seu principal objetivo é identificar o atleta mais próximo da bola durante a execução de jogadas de ataque e extrair a sua pose corporal por meio de landmarks.

Para viabilizar o processamento, o **OpenCV** é utilizado no gerenciamento e leitura dos vídeos/frames, a rede neural **Ultralytics YOLOv8** é aplicada para a detecção de pessoas e da bola em tempo real, e a API **MediaPipe Pose Landmarker** é empregada para reconstruir e mapear os pontos e conexões do esqueleto do atleta em destaque.

---

## Tecnologias utilizadas

* **Python 3.12+**
* **OpenCV (`opencv-python`)** — Leitura, manipulação de imagens/crop e renderização de saída dos vídeos.
* **MediaPipe (`mediapipe`)** — Detecção avançada de pose e extração dos landmarks corporais (`pose_landmarker_heavy.task`).
* **Ultralytics YOLOv8 (`yolov8n.pt`)** — Detecção multiobjeto por frame (classes: pessoa e bola).
* **NumPy** — Manipulação de matrizes de imagens e cálculos numéricos.

---

## Requisitos

* Python 3.10 ou superior instalado.
* Gerenciador de pacotes `pip`.
* Arquivo de modelo do MediaPipe: `src/pose_landmarker_heavy.task`.
* Pesos do modelo YOLO: `yolov8n.pt`.




## Estrutura básica do projeto

```text
SpikeAI/
├── input/                             # Diretório contendo os vídeos de entrada (.mp4)
├── output/                            # Diretório com os vídeos gerados e anotados
├── src/
│   ├── main.py                        # Ponto de entrada (executa a função principal)
│   ├── redender.py                    # Loop principal: orquestra leitura -> YOLO -> crop -> MediaPipe -> 
│   ├── video.py                       # Utilitários de manipulação de vídeo e lote (OpenCV VideoWriter)
│   ├── pose.py                        # Inicializador do MediaPipe Pose Landmarker
│   ├── yolo.py                        # Carregamento do modelo YOLOv8 e inferência por frame
│   └── pose_landmarker_heavy.task     # Modelo de IA treinado para pose do MediaPipe
├── yolov8n.pt                         # Pesos do modelo YOLOv8
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Documentação do projeto

---

## Fluxo de Funcionamento:

* O script `src/main.py` aciona o executor no `src/redender.py`.
* O `src/video.py` mapeia todos os vídeos contidos em `input/`.
* Para cada frame, o `src/yolo.py` faz a detecção das pessoas e da bola.
* O `src/redender.py` calcula qual pessoa está mais próxima da bola através do centroide de suas caixas delimitadoras (*bounding boxes*).
* O recorte (*crop*) correspondente ao atleta é passado para o detector de poses em `src/pose.py`.
* O esqueleto (conexões entre pontos) é desenhado sobre o vídeo e exportado para `output/`.

---

## Exemplo de entrada

Um ou mais vídeos de partidas ou treinos de vôlei posicionados na pasta `input/` (exemplo: `input/ataque.mp4`), contendo quadra, bola e atletas.

## Exemplo de saída

Um vídeo resultante na pasta `output/` (exemplo: `output/ataque_pose.mp4`) mantendo a taxa de quadros (FPS) e resolução original, apresentando:
* Uma caixa delimitadora indicando o atleta selecionado em foco.
* A renderização em tempo real das conexões de pose (landmarks) sobrepostas no jogador mais próximo da bola.

---

## Dificuldades e limitações encontradas

O projeto busca ir além de simples rastreadores de pose que analisam apenas uma pessoa isolada na imagem. Para focar a análise de movimento especificamente no atleta realizando o ataque, tornou-se fundamental combinar o rastreamento de bola do YOLO com o refinamento anatômico do MediaPipe.
Apesar do rastreio da bola, o atleta que deveria ser analisado acaba sendo trocado se tiver mais de uma pessoa no vídeo

### **Desafios no Cálculo de Proximidade:**
1. **Diferença Direta de Coordenadas:** A tentativa inicial de subtrair as coordenadas dos cantos das bounding boxes gerava problemas com valores negativos e distorcia o resultado dependendo do tamanho das caixas.
2. **Vetor Quadridimensional:** Tratar a bounding box `(x1, y1, x2, y2)` como um vetor 4D falhou pois o tamanho/proporção do retângulo da pessoa acabava sendo confundido com a distância física real até a bola.
3. **Solução Adotada (Distância Euclidiana):** A solução foi calcular a **Distância Euclidiana** entre os pontos centrais (centroides) das caixas da pessoa e da bola, reduzindo o cálculo para 2D.

### **Limitações Atuais:**
* **Oclusão e Velocidade da Bola:** Em lances rápidos de ataque, a bola de vôlei sofre desfoque de movimento (*motion blur*) ou oclusão, dificultando sua detecção contínua pelo YOLO.
* **Persistência de Rastreamento:** Como alternativa de contorno (*fallback*), quando a bola deixa de ser detectada temporariamente em alguns quadros, o sistema mantém o foco no último jogador identificado ou seleciona a maior detecção disponível para evitar quedas abruptas no desenho da pose.