"""OCR de placa (Mercosul LLLNLNN e antiga LLLNNNN) para o pipeline de video.

Dois motores de leitura, escolhidos em get_reader():
  - "fast"    : fast-plate-ocr, modelo especifico de placas   (padrao)
  - "easyocr" : EasyOCR, OCR generico (so como alternativa)

Uso no app.py:
    from ocr_placa import get_reader, ler_placa, VotadorPlaca, casar_cadastro, nitidez_crop
"""
import os
import re
import tempfile
from collections import defaultdict

import cv2

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Confusoes comuns do OCR (genericas, nao presas a uma placa especifica)
LETRA_PARA_NUM = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                  "E": "3", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
NUM_PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "3": "E", "4": "A", "5": "S",
                  "6": "G", "7": "T", "8": "B", "9": "P"}

REGEX_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")   # ABC1D23
REGEX_ANTIGA = re.compile(r"^[A-Z]{3}[0-9]{4}$")               # ABC1234

MODELO_FAST = "cct-s-v2-global-model"


def get_reader(backend="fast", gpu=None):
    """Cria o leitor de placas. No Streamlit, chame dentro de st.cache_resource."""
    if backend == "fast":
        from fast_plate_ocr import LicensePlateRecognizer
        return LicensePlateRecognizer(MODELO_FAST)

    import easyocr
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    return easyocr.Reader(["en"], gpu=gpu)


def formato_de(placa):
    """'mercosul' se a posicao 4 for letra, 'antiga' se for numero."""
    return "antiga" if placa[4].isdigit() else "mercosul"


def ajustar_placa(texto):
    """Normaliza o texto lido para LLLNLNN ou LLLNNNN. Retorna None se invalido.

    Posicoes 0-2 sao letras e 3, 5, 6 sao numeros nos dois formatos.
    A posicao 4 decide o formato: letra = Mercosul, numero = antiga.
    """
    clean = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(clean) < 7:
        return None
    c = list(clean[:7])
    for i in (0, 1, 2):
        c[i] = NUM_PARA_LETRA.get(c[i], c[i])
    for i in (3, 5, 6):
        c[i] = LETRA_PARA_NUM.get(c[i], c[i])
    placa = "".join(c)
    if REGEX_MERCOSUL.match(placa) or REGEX_ANTIGA.match(placa):
        return placa
    return None


# ---------------------------------------------------------------- recortes

def recortar_placa(frame, bbox, pad_x=0.05, pad_y=0.10):
    """Recorte colorido da placa, com uma folga pequena. Retorna None se vazio."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_x)); x2 = min(w, int(x2 + bw * pad_x))
    y1 = max(0, int(y1 - bh * pad_y)); y2 = min(h, int(y2 + bh * pad_y))
    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


def preparar_crop(frame, bbox, pad_x=0.10, pad_y=0.20, escala=3, clahe=True):
    """Pre-processamento para o EasyOCR: amplia 3x, realca contraste e suaviza."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_x)); x2 = min(w, int(x2 + bw * pad_x))
    y1 = max(0, int(y1 - bh * pad_y)); y2 = min(h, int(y2 + bh * pad_y))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    if clahe:
        cinza = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cinza = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cinza)
        return cv2.bilateralFilter(cinza, 9, 75, 75)
    return cv2.bilateralFilter(crop, 9, 75, 75)


# ---------------------------------------------------------------- leitura

# A biblioteca le o arquivo e cuida sozinha da conversao de cor exigida pelo modelo.
_TMP = os.path.join(tempfile.gettempdir(), f"granorte_placa_{os.getpid()}.png")


def _ler_fast(reader, frame, bbox):
    crop = recortar_placa(frame, bbox)
    if crop is None:
        return None, 0.0, "", None
    cv2.imwrite(_TMP, crop)
    preds = reader.run(_TMP, return_confidence=True)
    if not preds:
        return None, 0.0, "", crop
    pred = preds[0]
    texto = (pred.plate or "").strip("_ ")
    if not texto:
        return None, 0.0, "", crop
    probs = [float(p) for p in pred.char_probs][:len(texto)]
    conf = sum(probs) / len(probs) if probs else 0.0
    return ajustar_placa(texto), conf, texto, crop


def _ler_easyocr(reader, frame, bbox, clahe=True):
    img = preparar_crop(frame, bbox, clahe=clahe)
    if img is None:
        return None, 0.0, "", None
    res = reader.readtext(img, allowlist=ALLOWLIST, detail=1)
    if not res:
        return None, 0.0, "", img
    res = sorted(res, key=lambda r: r[0][0][0])        # esquerda -> direita
    bruto = "".join(r[1] for r in res)
    conf = sum(r[2] for r in res) / len(res)
    return ajustar_placa(bruto), float(conf), bruto, img


def ler_placa(reader, frame, bbox, clahe=True):
    """Retorna (placa|None, confianca, texto_bruto, imagem_do_recorte).

    Escolhe o motor pelo tipo de `reader` (o do get_reader())."""
    if hasattr(reader, "readtext"):                     # EasyOCR
        return _ler_easyocr(reader, frame, bbox, clahe)
    return _ler_fast(reader, frame, bbox)               # fast-plate-ocr


# ---------------------------------------------------------------- votacao

class VotadorPlaca:
    """Voto ponderado por caractere, separado por formato de placa.
    Leituras com peso maior (recorte mais nitido, OCR mais confiante) mandam mais."""

    def __init__(self):
        self.votos = {f: [defaultdict(float) for _ in range(7)]
                      for f in ("mercosul", "antiga")}
        self.peso_formato = defaultdict(float)
        self.leituras = 0

    def add(self, placa, peso=1.0):
        if not placa:
            return
        f = formato_de(placa)
        for i, ch in enumerate(placa):
            self.votos[f][i][ch] += peso
        self.peso_formato[f] += peso
        self.leituras += 1

    def _formato(self):
        return max(self.peso_formato, key=self.peso_formato.get)

    def melhor(self, minimo=5):
        """So devolve resultado depois de `minimo` leituras validas."""
        if self.leituras < minimo:
            return None
        v = self.votos[self._formato()]
        return "".join(max(pos, key=pos.get) for pos in v)

    def confianca(self):
        """0 a 1: media de quanto o vencedor domina cada posicao."""
        if not self.leituras:
            return 0.0
        v = self.votos[self._formato()]
        return sum(max(p.values()) / sum(p.values()) for p in v) / 7

    def reset(self):
        self.__init__()


# ---------------------------------------------------------------- cadastro

def _distancia(a, b):
    """Distancia de edicao (Levenshtein), sem dependencias."""
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1, ant[j - 1] + (ca != cb)))
        ant = atual
    return ant[-1]


def casar_cadastro(placa, cadastro, max_dist=1):
    """Compara a placa lida com as placas cadastradas.

    Retorna (placa_cadastrada, distancia) ou (None, None) se nenhuma estiver
    dentro de `max_dist` ou se houver empate entre duas (ambiguo)."""
    if not placa or not cadastro:
        return None, None
    ranking = sorted((_distancia(placa, p.upper()), p.upper()) for p in cadastro)
    d, p = ranking[0]
    if d > max_dist:
        return None, None
    if len(ranking) > 1 and ranking[1][0] == d:
        return None, None
    return p, d


# ---------------------------------------------------------------- qualidade

def nitidez_crop(frame, bbox):
    """Nitidez do recorte da placa (variancia do Laplaciano) no tamanho original.
    Frames borrados ou estourados pelo farol dao valores baixos."""
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    cinza = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())
