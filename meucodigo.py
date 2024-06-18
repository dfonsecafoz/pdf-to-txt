import os
import faiss
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from openai import OpenAI
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Obter a chave de API da OpenAI do arquivo .env
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def save_text_to_file(text, txt_path):
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text)

def load_index(index_path):
    return faiss.read_index(index_path)

def search_index(index, query_embedding, k=5):
    D, I = index.search(query_embedding, k)  # D são as distâncias, I são os índices dos vetores mais próximos.
    return D, I

# Função para obter o embedding usando a API da OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def main():
    st.title("Busca de Similaridade em PDFs")
    uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")

    if uploaded_file is not None:
        # Parâmetros ajustáveis
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)

        # Salvar o arquivo carregado em um arquivo temporário
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        txt_path = 'meucodigo.txt'
        chunks_dir = './chunks'
        index_path = './faiss_index.index'

        # Criar o diretório /chunks se não existir
        if not os.path.exists(chunks_dir):
            os.makedirs(chunks_dir)

        # Carregar o PDF e dividir em páginas usando PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # tamanho do chunk
            chunk_overlap=chunk_overlap  # sobreposição entre chunks
        ))

        chunks = []
        chunk_references = []

        # Processar os documentos carregados
        for doc in documents:
            chunk = doc.page_content
            pages = doc.metadata.get('page', 'desconhecida')
            # Adicionar 1 à página para corrigir a referência
            if isinstance(pages, int):
                pages += 1
            chunk_references.append((pages, 0, len(chunk)))  # Aqui, 0 e len(chunk) são placeholders
            chunks.append(chunk)

        # Gerar embeddings para os chunks usando o modelo da OpenAI
        embeddings = np.array([get_embedding(chunk) for chunk in chunks]).astype('float32')

        # Criar e treinar o índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Salvar o índice FAISS
        faiss.write_index(index, index_path)

        # Salvar os chunks e os embeddings no diretório /chunks
        for i, (chunk, embedding, reference) in enumerate(zip(chunks, embeddings, chunk_references)):
            chunk_with_reference = f"Pages: {reference[0]}, Start: {reference[1]}, End: {reference[2]}\n{chunk}"
            chunk_txt_path = os.path.join(chunks_dir, f'chunk_{i+1}.txt')
            save_text_to_file(chunk_with_reference, chunk_txt_path)
            
            embedding_txt_path = os.path.join(chunks_dir, f'embedding_{i+1}.txt')
            save_text_to_file(str(embedding), embedding_txt_path)

        # Opcional: salvar o texto completo em um único arquivo
        full_text = "\n".join(chunks)
        save_text_to_file(full_text, txt_path)

        st.success("Processamento do PDF concluído! Você pode agora fazer uma pergunta.")

        # Permitir ao usuário inserir uma consulta de texto após o PDF ter sido processado
        query_text = st.text_input("Digite sua consulta:")

        if query_text:
            # Carregar o índice FAISS
            index = load_index(index_path)

            # Gerar embedding para a consulta usando o modelo da OpenAI
            query_embedding = np.array([get_embedding(query_text)]).astype('float32')

            # Realizar a busca no índice FAISS
            D, I = search_index(index, query_embedding)

            # Exibir os resultados
            st.write("Distâncias:", D)
            st.write("Índices dos vetores mais próximos:", I)

            for idx in I[0]:  # Iterar pelos índices dos vetores mais próximos
                st.write(f'Chunk encontrado: {chunks[idx]}')

if __name__ == '__main__':
    main()
