import re
import string
import nltk
from transformers import pipeline

# Initialize zero-shot classifier once at module import
classifier = pipeline("zero-shot-classification", 
                     model="joeddav/xlm-roberta-large-xnli")

# Get Portuguese stopwords
try:
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
except LookupError:
    # Fallback if stopwords not downloaded
    stopwords = set()

def preprocess_text(text):
    """Clean and preprocess text for classification."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords]
    
    return ' '.join(words)

def _has_ids(text):
    """Check if text contains ticket numbers, CPF, or CNPJ patterns."""
    # Look for common patterns: numbers, CPF (xxx.xxx.xxx-xx), CNPJ
    ticket_pattern = r'\b\d{4,}\b'  # 4+ digit numbers
    cpf_pattern = r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'
    cnpj_pattern = r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'
    
    return bool(re.search(ticket_pattern, text) or 
                re.search(cpf_pattern, text) or 
                re.search(cnpj_pattern, text))

def classify_email(text):
    """Classify email as Produtivo or Improdutivo using zero-shot classification."""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Classify with zero-shot
    result = classifier(processed_text, 
                       candidate_labels=["Produtivo", "Improdutivo"])
    
    label = result['labels'][0]
    score = result['scores'][0]
    
    return label, score

def suggest_reply(text, label, score, low=0.6):
    """Generate appropriate reply based on classification and confidence."""
    
    # Low confidence - use generic safe reply
    if score < low:
        return ("Olá,\n\n"
                "Obrigado pelo seu contato. Para que possamos atendê-lo da melhor forma, "
                "por favor, forneça o número do seu ticket, CPF ou CNPJ.\n\n"
                "Atenciosamente,\n"
                "Equipe de Atendimento")
    
    # Improdutivo - polite thanks and close
    if label == "Improdutivo":
        return ("Olá,\n\n"
                "Obrigado pelo seu contato e pelas suas palavras.\n\n"
                "Atenciosamente,\n"
                "Equipe de Atendimento")
    
    # Produtivo - check for IDs and respond accordingly
    if label == "Produtivo":
        if _has_ids(text):
            return ("Olá,\n\n"
                    "Recebemos sua solicitação e estamos verificando as informações. "
                    "Retornaremos com uma atualização até o final do expediente.\n\n"
                    "Atenciosamente,\n"
                    "Equipe de Atendimento")
        else:
            return ("Olá,\n\n"
                    "Obrigado pelo seu contato. Para que possamos processar sua solicitação, "
                    "por favor, forneça o número do seu ticket, CPF ou CNPJ.\n\n"
                    "Atenciosamente,\n"
                    "Equipe de Atendimento")
    
    # Fallback
    return ("Olá,\n\n"
            "Obrigado pelo seu contato. Retornaremos em breve.\n\n"
            "Atenciosamente,\n"
            "Equipe de Atendimento")
