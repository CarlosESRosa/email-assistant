from flask import Flask, render_template, request, flash
import os
import nltk

app = Flask(__name__)
app.secret_key = 'autou-mail-assistant-secret-key'

# Download NLTK stopwords on first run
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from utils.nlp import classify_email, suggest_reply
from utils.parsers import read_text_from_upload

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get text from textarea or uploaded file
        text = request.form.get('email_text', '').strip()
        uploaded_file = request.files.get('email_file')
        
        # Validate input
        if not text and not uploaded_file:
            flash('Por favor, cole o texto do e-mail ou faça upload de um arquivo.', 'error')
            return render_template('index.html')
        
        # Read text from file if uploaded
        if uploaded_file and uploaded_file.filename:
            file_text = read_text_from_upload(uploaded_file)
            if file_text:
                text = file_text
            else:
                flash('Erro ao processar o arquivo. Verifique se é um .txt ou .pdf válido.', 'error')
                return render_template('index.html')
        
        if not text.strip():
            flash('Nenhum texto válido encontrado.', 'error')
            return render_template('index.html')
        
        # Classify email
        label, score = classify_email(text)
        
        # Generate reply
        reply = suggest_reply(text, label, score)
        
        return render_template('result.html', 
                             text=text, 
                             label=label, 
                             score=round(score, 2), 
                             reply=reply)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
