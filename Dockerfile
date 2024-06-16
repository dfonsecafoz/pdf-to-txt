# Usar a imagem base do langchain
FROM langchain/langchain:latest

# Instalar PyPDF2, sentence-transformers, pypdf, faiss e streamlit
RUN pip install PyPDF2 sentence-transformers pypdf faiss-cpu streamlit

# Copiar o código para o contêiner
COPY meucodigo.py /app/meucodigo.py

# Definir o diretório de trabalho
WORKDIR /app

# Executar o Streamlit
CMD ["streamlit", "run", "meucodigo.py", "--server.port=8501", "--server.address=0.0.0.0"]
