# AutoU - Classificador de E-mails

Um classificador simples de e-mails que distingue entre mensagens **Produtivas** e **Improdutivas** e sugere respostas automáticas em português.

## Funcionalidades

- Classificação de e-mails usando zero-shot learning (XLM-RoBERTa)
- Suporte a upload de arquivos (.txt, .pdf) ou entrada de texto
- Geração de respostas automáticas baseadas em regras
- Interface limpa e responsiva com Tailwind CSS
- Processamento local - nenhum dado é salvo

## Como executar localmente

1. **Criar ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Baixar stopwords do NLTK:**
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Executar aplicação:**
   ```bash
   flask run
   ```

Acesse `http://localhost:5000` no seu navegador.

## Deploy no Render

1. Conecte seu repositório ao Render
2. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
3. Deploy!

## Uso

1. Cole o texto do e-mail na caixa de texto OU faça upload de um arquivo (.txt ou .pdf)
2. Clique em "Processar"
3. Veja a classificação e a resposta sugerida
4. Use o botão "Copiar Resposta" para copiar a resposta gerada

## Limitações

- Primeira execução pode demorar para carregar o modelo
- Classificação baseada em regras simples
- Não salva dados - processamento local apenas
- Requer conexão com internet para carregar o modelo

## Exemplos

Veja os arquivos de exemplo em `sample_emails/` para testar a classificação.
