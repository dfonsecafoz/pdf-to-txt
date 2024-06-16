# Usar a imagem base do langchain
FROM langchain/langchain:latest

# Instalar PyPDF2, sentence-transformers, pypdf e faiss
RUN pip install PyPDF2 sentence-transformers pypdf faiss-cpu

# Copiar o código para o contêiner
COPY meucodigo.py /app/meucodigo.py

# Definir o diretório de trabalho
WORKDIR /app

# Executar o código
CMD ["python", "meucodigo.py"]
