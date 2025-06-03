import json
import os
import string
from typing import Optional
from pathlib import Path

import nltk
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from openai import OpenAIError
from pydantic import BaseModel


# -------------------------
# Configuração de diretórios
# -------------------------
BASE_DIR = Path(__file__).parent              # /caminho/para/meu_projeto
STATIC_DIR = BASE_DIR / "static"               # /caminho/para/meu_projeto/static

# -------------------------
# Carregar variável de ambiente (API Key)
# -------------------------

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise RuntimeError("Faltou definir a variável de ambiente OPENAI_API_KEY no teu .env")

# Inicialização NLTK (download só de stopwords)
nltk.download("stopwords")
STOPWORDS_PT = set(stopwords.words("portuguese"))
TOKENIZER = RegexpTokenizer(r"\w+")

# FastAPI app
app = FastAPI(title="API de Análise de Sentimento")


# ----------------------------------------------------------
# Funções internas de pré-processamento e chamada ao ChatGPT
# ----------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    1) Converte para minúsculas.
    2) Remove pontuação.
    3) Tokeniza usando RegexpTokenizer.
    4) Remove stopwords.
    5) Reconstrói a string apenas com tokens úteis.
    """
    texto_minusculo = text.lower()
    sem_pontuacao = texto_minusculo.translate(str.maketrans("", "", string.punctuation))
    tokens = TOKENIZER.tokenize(sem_pontuacao)
    tokens_limpos = [t for t in tokens if t not in STOPWORDS_PT]
    return " ".join(tokens_limpos)


def call_chatgpt_for_sentiment(preprocessed: str) -> dict:
    """
    Envia o texto pré-processado ao ChatGPT, pedindo-lhe um JSON com
    'neg', 'neu', 'pos' e 'compound'. Retorna dict ou lança exceção.
    """
    prompt = (
        "Avalia o sentimento do texto abaixo (já pré-processado) e devolve apenas uma string JSON "
        "com quatro campos exatos: 'neg', 'neu', 'pos' e 'compound'. "
        "Para 'neg', 'neu' e 'pos', usa valores de 0.0 a 1.0; para 'compound', usa -1.0 a 1.0, "
        "como no VADER (NLTK).\n\n"
        f"Texto pré-processado:\n\"{preprocessed}\"\n\n"
        "Retorna somente o JSON, sem texto adicional."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "És um avaliador de sentimentos."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        # Tenta converter a string JSON em dict
        return json.loads(content)

    except OpenAIError as e:
        msg = str(e)
        if "Rate limit" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=503,
                                detail="Quota esgotada (RateLimit). Verifica no painel da OpenAI se ainda tens créditos disponíveis.")
        if "Invalid" in msg or "Authentication" in msg:
            raise HTTPException(status_code=401, detail="Erro de autenticação: verifica a tua chave API da OpenAI.")
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API da OpenAI: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Não foi possível interpretar a resposta do ChatGPT como JSON.")


# --------------------------------------------
# Modelos Pydantic para validação de input/output
# --------------------------------------------

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float
    # Se quiseres incluir um campo opcional de erro (nunca ambos simultaneamente):
    error: Optional[str] = None


# --------------------------
# Endpoints da API
# --------------------------

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(req: SentimentRequest):
    """
    Endpoint que recebe um JSON com {"text": "..."} e devolve {"neg": ..., "neu": ..., "pos": ..., "compound": ...}.
    """
    texto_original = req.text.strip()
    if not texto_original:
        raise HTTPException(status_code=400, detail="O campo 'text' não pode estar vazio.")

    # 1) Pré-processar
    texto_limpo = preprocess_text(texto_original)

    # 2) Chamar ChatGPT e obter JSON
    resultado = call_chatgpt_for_sentiment(texto_limpo)

    # 3) Validar que o JSON retornado contém as chaves necessárias
    for chave in ("neg", "neu", "pos", "compound"):
        if chave not in resultado:
            raise HTTPException(status_code=502, detail=f"Resposta inesperada do ChatGPT: faltou a chave '{chave}'.")

    # 4) Converter valores para float e devolver
    try:
        return SentimentResponse(
            neg=float(resultado["neg"]),
            neu=float(resultado["neu"]),
            pos=float(resultado["pos"]),
            compound=float(resultado["compound"]),
        )
    except (ValueError, TypeError):
        raise HTTPException(status_code=502, detail="Valores inválidos no JSON devolvido pelo ChatGPT.")

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    return FileResponse(f"static/{file_path}")
