"""Testa o OCR especifico de placas (fast-plate-ocr) numa imagem de placa recortada.

Instalacao (uma vez, com o .venv ativado):
    pip install "fast-plate-ocr[onnx]"

Uso (na raiz do projeto):
    python testar_fast_plate.py placa.png

Na primeira execucao o modelo e baixado da internet.
"""
import sys

from fast_plate_ocr import LicensePlateRecognizer

if len(sys.argv) < 2:
    sys.exit("Uso: python testar_fast_plate.py <imagem_da_placa>")

reconhecedor = LicensePlateRecognizer("cct-s-v2-global-model")
pred = reconhecedor.run(sys.argv[1], return_confidence=True)[0]

print(f"Placa lida          : {pred.plate}")
print(f"Confianca por letra : {pred.char_probs}")
