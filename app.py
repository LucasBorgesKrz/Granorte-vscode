import os, glob, subprocess

import cv2
import streamlit as st
from ultralytics import YOLO
from ocr_placa import get_reader, ler_placa, VotadorPlaca, casar_cadastro, nitidez_crop
CADASTRO = ["SNC4G97", "OJI1239"]



VIDEOS_DIR = "videos"
MODELO_PATH = "modelo/best.pt"
SAIDA_DIR = "saida"

os.makedirs(SAIDA_DIR, exist_ok=True)
DEBUG_DIR = "saida/debug_placas"
os.makedirs(DEBUG_DIR, exist_ok=True)

st.set_page_config(page_title="Visao Granorte IA - Deteccao de Placa", layout="wide")
st.title("Deteccao de Placa Veicular")

if not os.path.exists(MODELO_PATH):
    st.error(
        f"Modelo nao encontrado em '{MODELO_PATH}'.\n\n"
        "Baixe o 'best.pt' gerado no Colab (treino_modelo_placa.ipynb) "
        "e coloque nessa pasta com esse nome."
    )
    st.stop()

@st.cache_resource
def carregar_modelo():
    return YOLO(MODELO_PATH)
@st.cache_resource
def carregar_reader():
    return get_reader()

videos = sorted(glob.glob(f"{VIDEOS_DIR}/*.mp4"))
if not videos:
    st.warning(f"Nenhum video .mp4 encontrado em '{VIDEOS_DIR}/'. Coloque seus videos nessa pasta.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    escolhido = st.selectbox("Video", videos, format_func=os.path.basename)
with col2:
    limiar = st.slider("Confianca minima", 0.1, 0.9, 0.25)

nome = os.path.splitext(os.path.basename(escolhido))[0]
saida_final = f"{SAIDA_DIR}/{nome}_deteccao.mp4"

if st.button("Processar video"):
    model = carregar_modelo()
    reader = carregar_reader()
    votador = VotadorPlaca()
    placas_lidas = []
    sem_placa = 0
    barra = st.progress(0, text="Processando frames...")
    
    cap = cv2.VideoCapture(escolhido)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bruto = f"{SAIDA_DIR}/{nome}_bruto.mp4"
    writer = cv2.VideoWriter(bruto, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=limiar, verbose=False)[0]

        # Sem placa por ~2s: fecha a leitura atual (vale para vídeos com mais de um caminhão)
        if len(r.boxes) == 0:
            sem_placa += 1
            if sem_placa > fps * 2:
                if votador.melhor():
                    placas_lidas.append(votador.melhor())
                votador.reset()
        else:
            sem_placa = 0

        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])

            # OCR a cada 3 frames, só com placa confiável e grande o bastante
            if conf >= 0.5 and (x2 - x1) >= 40 and i % 2 == 0:
                placa, c_ocr, texto_ocr, img_dbg = ler_placa(reader, frame, (x1, y1, x2, y2))
                nit = nitidez_crop(frame, (x1, y1, x2, y2))
                if placa and c_ocr >= 0.10:
                    votador.add(placa, peso=c_ocr * nit ** 2)   
                if img_dbg is not None:
                    cv2.imwrite(f"{DEBUG_DIR}/{nome}_f{i}_{texto_ocr}_{c_ocr:.2f}_n{nit:.0f}.jpg", img_dbg)

            texto = votador.melhor() or "lendo..."
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{texto} ({conf:.2f})", (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(frame)
        i += 1
        barra.progress(min(i / max(total, 1), 1.0), text=f"Frame {i}/{total}")

    cap.release()
    writer.release()

    if votador.melhor():              # última placa do vídeo
        placas_lidas.append(votador.melhor())

    st.info("Convertendo video para o navegador reproduzir...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", bruto,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", saida_final],
            check=True,
        )
        os.remove(bruto)
    except FileNotFoundError:
        st.warning("ffmpeg nao encontrado no PATH — exibindo o video sem reconversao.")
        saida_final = bruto

    st.success("Deteccao concluida.")
    if placas_lidas:
        for p in placas_lidas:
            cad, dist = casar_cadastro(p, CADASTRO, max_dist=1)
            if cad:
                st.success(f"Lida: {p}  →  cadastrada: {cad} (diferença de {dist} caractere(s))")
            else:
                st.warning(f"Lida: {p}  →  NÃO cadastrada (confirmar manualmente)")
    else:
        st.write("Nenhuma placa foi lida com confiança.")

if os.path.exists(saida_final):
    st.video(saida_final)
