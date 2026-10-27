"""Extrai frames de um video para montar o dataset de treino do YOLO.

Uso (na raiz do projeto, com o .venv ativado):
    python extrair_frames.py videos/carro.mp4 frames_carro 5
O ultimo numero e opcional: salva 1 frame a cada N (padrao 5).
"""
import os
import sys

import cv2

if len(sys.argv) < 3:
    sys.exit("Uso: python extrair_frames.py <video.mp4> <pasta_saida> [a_cada_N_frames]")

video, saida = sys.argv[1], sys.argv[2]
passo = int(sys.argv[3]) if len(sys.argv) > 3 else 5

os.makedirs(saida, exist_ok=True)
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"Nao consegui abrir o video: {video}")

nome = os.path.splitext(os.path.basename(video))[0]
i = salvos = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % passo == 0:
        cv2.imwrite(f"{saida}/{nome}_{i:05d}.jpg", frame)
        salvos += 1
    i += 1
cap.release()
print(f"{salvos} frames salvos em '{saida}' (de {i} lidos)")
