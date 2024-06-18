from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import numpy as np

# Configure sua chave de API da OpenAI
client = OpenAI(api_key='sk-proj-p1eGorAXHgP9y90wanlOT3BlbkFJKhbZlw9SRxJuRCXooa2N')

# Função para obter o embedding usando a API da OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Função para calcular a similaridade de cosseno entre dois vetores
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Função para carregar e dividir o PDF em chunks de texto
def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        for start in range(0, len(text), chunk_size - chunk_overlap):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({'page': page_num + 1, 'content': chunk})
    return chunks

# Função para calcular embeddings e adicionar ao DataFrame
def calculate_embeddings(chunks, model="text-embedding-3-small"):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk['content'], model=model)
        embeddings.append(embedding)
    return embeddings

# Função para buscar reviews semelhantes
def search_reviews(df, product_description, n=3, model="text-embedding-3-small"):
    embedding = get_embedding(product_description, model=model)
    df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

# Caminho para o arquivo PDF
pdf_path = 'seu_arquivo.pdf'  # Substitua pelo caminho do seu arquivo PDF

# Carregar e dividir o PDF em chunks
chunks = load_and_split_pdf(pdf_path)

# Criar um DataFrame com os chunks
df = pd.DataFrame(chunks)

# Calcular embeddings para cada chunk
df['embedding'] = calculate_embeddings(chunks)

# Descrição do produto para a busca
product_description = 'delicious beans'

# Buscar os chunks mais semelhantes
result = search_reviews(df, product_description, n=3)

# Exibir os resultados
for index, row in result.iterrows():
    print(f"Page: {row['page']}, Similarity: {row['similarities']}")
    print(f"Content: {row['content']}\n")
