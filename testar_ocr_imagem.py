"""Testa so o OCR numa imagem de placa ja recortada (sem o detector YOLO).

Uso (na raiz do projeto, com o .venv ativado):
    python testar_ocr_imagem.py placa_carro.png
"""
import sys

import cv2

from ocr_placa import get_reader, ler_placa

if len(sys.argv) < 2:
    sys.exit("Uso: python testar_ocr_imagem.py <imagem_da_placa>")

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit(f"Nao consegui abrir a imagem: {sys.argv[1]}")

h, w = img.shape[:2]
placa, conf, bruto, _ = ler_placa(get_reader(), img, (0, 0, w, h))
print(f"Texto bruto do OCR : {bruto!r}")
print(f"Placa normalizada  : {placa}")
print(f"Confianca do OCR   : {conf:.2f}")
