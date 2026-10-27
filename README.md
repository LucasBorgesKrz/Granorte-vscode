# Visao Granorte IA - Deteccao local (Streamlit)

Roda a deteccao de placa nos videos usando o modelo YOLOv8 treinado no Google Colab
(notebook `treino_modelo_placa.ipynb`). Este projeto NAO treina nada, NAO acessa o
Roboflow e NAO faz OCR — so carrega o `best.pt` ja pronto e desenha as deteccoes no video.

## Passo 1 - Treinar no Colab

1. Abra `treino_modelo_placa.ipynb` no Google Colab.
2. Ative a GPU (T4) e configure o secret `ROBOFLOW_API_KEY`.
3. Rode todas as celulas. No final, ele baixa `modelo_placa.zip`.
4. Extraia o zip. Dentro dele esta o `best.pt`.

## Passo 2 - Configurar o projeto local

1. Copie o `best.pt` para a pasta `modelo/` deste projeto (deve ficar em `modelo/best.pt`).
2. Copie os videos (`teste1.mp4`, `teste2.mp4`, etc.) para a pasta `videos/`.

## Passo 3 - Preparar o ambiente

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Instale tambem o ffmpeg (usado para o video ficar reproduzivel no navegador):
- Windows: baixe em ffmpeg.org e adicione a pasta bin/ ao PATH
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

Confirme com:
```bash
ffmpeg -version
```

## Passo 4 - Rodar o app

```bash
streamlit run app.py
```

Abre em http://localhost:8501. Escolha o video, ajuste a confianca minima e clique em
"Processar video" para ver a deteccao rodando.

## Estrutura

```
granorte-deteccao-local/
├── app.py
├── requirements.txt
├── modelo/
│   └── best.pt        <- vem do Colab
├── videos/
│   └── teste1.mp4      <- seus videos
└── saida/               <- criado automaticamente com os resultados
```
