import os
import faiss
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
import redis
import hashlib
from openai import OpenAI

client = OpenAI()

def save_text_to_file(text, txt_path):
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text)

def save_index_to_redis(index, redis_client, key):
    index_bytes = faiss.serialize_index(index)  # Serialize FAISS index
    redis_client.set(key, index_bytes.tobytes())

def load_index_from_redis(redis_client, key):
    index_bytes = redis_client.get(key)
    if index_bytes is not None:
        index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
        return index
    else:
        return None

def search_index(index, query_embedding, k=5):
    D, I = index.search(query_embedding, k)  # D são as distâncias, I são os índices dos vetores mais próximos
    return D, I

def get_combined_key(file_paths):
    hasher = hashlib.md5()
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
    return hasher.hexdigest()

def get_openai_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def main():
    st.title("Busca de Similaridade em PDFs")
    uploaded_files = st.file_uploader("Envie seus PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Parâmetros ajustáveis
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)

        # Conexão com o Redis
        redis_client = redis.StrictRedis(host=os.environ.get("REDIS_HOST", "localhost"), port=int(os.environ.get("REDIS_PORT", 6379)))

        # Salvar os arquivos carregados em arquivos temporários
        file_paths = []
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_paths.append(tmp_file.name)

        txt_path = 'meucodigo.txt'
        chunks_dir = './chunks'
        index_path = './faiss_index.index'

        # Criar o diretório /chunks se não existir
        if not os.path.exists(chunks_dir):
            os.makedirs(chunks_dir)

        chunks = []
        chunk_references = []

        # Processar cada PDF
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,  # tamanho do chunk
                chunk_overlap=chunk_overlap  # sobreposição entre chunks
            ))

            for doc in documents:
                chunk = doc.page_content
                pages = doc.metadata.get('page', 'desconhecida')
                # Adicionar 1 à página para corrigir a referência
                if isinstance(pages, int):
                    pages += 1
                chunk_references.append((pages, 0, len(chunk)))  # Aqui, 0 e len(chunk) são placeholders
                chunks.append(chunk)

        # Gerar embeddings para os chunks usando a OpenAI
        embeddings = np.array([get_openai_embedding(chunk) for chunk in chunks]).astype('float32')

        # Tentar carregar o índice FAISS do Redis
        index_key = get_combined_key(file_paths)
        index = load_index_from_redis(redis_client, index_key)
        if index is None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
        index.reset()  # Certifique-se de que o índice esteja vazio antes de adicionar novos embeddings
        index.add(embeddings)
        save_index_to_redis(index, redis_client, index_key)

        # Salvar os chunks e os embeddings no diretório /chunks
        for i, (chunk, embedding, reference) in enumerate(zip(chunks, embeddings, chunk_references)):
            chunk_with_reference = f"Pages: {reference[0]}, Start: {reference[1]}, End: {reference[2]}\n{chunk}"
            chunk_txt_path = os.path.join(chunks_dir, f'chunk_{i+1}.txt')
            save_text_to_file(chunk_with_reference, chunk_txt_path)
            st.write(f'Chunk {i+1} salvo em {chunk_txt_path}')

            embedding_txt_path = os.path.join(chunks_dir, f'embedding_{i+1}.txt')
            save_text_to_file(str(embedding), embedding_txt_path)
            st.write(f'Embedding {i+1} salvo em {embedding_txt_path}')

        # Opcional: salvar o texto completo em um único arquivo
        full_text = "\n".join(chunks)
        save_text_to_file(full_text, txt_path)
        st.write(f'Texto completo salvo em {txt_path}')

        st.success("Processamento dos PDFs concluído!")

    query_text = st.text_input("Digite sua consulta:")

    if query_text:
        # Gerar a chave do índice FAISS com base nos PDFs carregados
        if uploaded_files:
            index_key = get_combined_key(file_paths)
        else:
            st.error("Por favor, envie PDFs primeiro.")
            return

        # Carregar o índice FAISS do Redis
        index = load_index_from_redis(redis_client, index_key)
        if index is not None:
            query_embedding = np.array([get_openai_embedding(query_text)]).astype('float32')
            D, I = search_index(index, query_embedding)
            st.write("Distâncias:", D)
            st.write("Índices dos vetores mais próximos:", I)
            for idx in I[0]:
                if idx < len(chunks):
                    st.write(f'Chunk encontrado: {chunks[idx]}')
                else:
                    st.write(f'Índice {idx} está fora do alcance dos chunks.')
        else:
            st.error("Índice FAISS não encontrado no Redis.")

if __name__ == '__main__':
    main()

