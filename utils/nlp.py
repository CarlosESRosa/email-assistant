import os
import re
import string
import nltk
import requests

# -----------------------------
# Configuração do Hugging Face
# -----------------------------
# Modelo zero-shot multilíngue leve (roda na nuvem)
HF_ENDPOINT = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# -----------------------------
# Stopwords PT-BR
# -----------------------------
try:
    STOPWORDS = set(nltk.corpus.stopwords.words("portuguese"))
except LookupError:
    STOPWORDS = set()

def preprocess_text(text: str) -> str:
    """Minimamente normaliza o texto (lower, sem pontuação, sem stopwords)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# -----------------------------
# Heurística simples (fallback)
# -----------------------------
def _heuristic_classify(text: str):
    t = text.lower()
    impro_signals = ["parabéns", "obrigado", "obrigada", "agradeço", "feliz", "boas festas"]
    if any(s in t for s in impro_signals):
        return "Improdutivo", 0.70
    return "Produtivo", 0.65

def _has_ids(text: str):
    """Vê se há ticket/CPF/CNPJ para personalizar a resposta."""
    ticket_pattern = r"\b\d{4,}\b"
    cpf_pattern = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
    cnpj_pattern = r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"
    return bool(re.search(ticket_pattern, text) or
                re.search(cpf_pattern, text) or
                re.search(cnpj_pattern, text))

# -----------------------------
# Classificação via HF Inference
# -----------------------------
def _zero_shot_api(text: str):
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["Produtivo", "Improdutivo"],
            "multi_label": False
        }
    }
    resp = requests.post(HF_ENDPOINT, headers=HF_HEADERS, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    # Formato típico: {"labels": [...], "scores": [...]}
    label = data["labels"][0]
    score = float(data["scores"][0])
    return label, score

def classify_email(text: str):
    """Tenta classificar via API; se falhar (sem token/limite), usa heurística."""
    processed = preprocess_text(text)
    try:
        return _zero_shot_api(processed)
    except Exception:
        return _heuristic_classify(text)

def suggest_reply(text: str, label: str, score: float, low: float = 0.6):
    """Gera resposta curta, segura e adequada à classe/nível de confiança."""
    if score < low:
        return ("Olá,\n\n"
                "Recebemos sua mensagem. Para agilizar o atendimento, por favor informe "
                "o número do chamado ou CPF/CNPJ vinculado.\n\n"
                "Atenciosamente,\nEquipe de Atendimento")

    if label == "Improdutivo":
        return ("Olá,\n\n"
                "Obrigado pela mensagem! Agradecemos o contato. "
                "Se precisar de algo relacionado aos seus serviços, é só responder este e-mail.\n\n"
                "Atenciosamente,\nEquipe de Atendimento")

    # Produtivo
    if _has_ids(text):
        return ("Olá,\n\n"
                "Obrigado pelo contato. Localizamos sua solicitação e já estamos verificando. "
                "Você receberá uma atualização até o final do expediente.\n\n"
                "Atenciosamente,\nEquipe de Atendimento")
    else:
        return ("Olá,\n\n"
                "Obrigado pelo contato. Para dar andamento, poderia confirmar o número do chamado "
                "ou CPF/CNPJ vinculado? Assim atualizamos o status para você o quanto antes.\n\n"
                "Atenciosamente,\nEquipe de Atendimento")
