import pdfplumber

def read_text_from_upload(file):
    """Extract text from uploaded file (.txt or .pdf)."""
    if not file or not file.filename:
        return ""
    
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.txt'):
            # Read text file with UTF-8 encoding, ignore errors
            content = file.read()
            return content.decode('utf-8', errors='ignore')
        
        elif filename.endswith('.pdf'):
            # Extract text from PDF using pdfplumber
            file.seek(0)  # Reset file pointer
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Handle empty pages safely
                        text += page_text + "\n"
            return text.strip()
        
        else:
            return ""
    
    except Exception:
        # Return empty string on any error
        return ""
