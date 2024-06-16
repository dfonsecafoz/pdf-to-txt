import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.pdf import PyPDFLoader

def save_text_to_file(text, txt_path):
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text)

def main():
    pdf_path = 'meucodigo.pdf'
    txt_path = 'meucodigo.txt'
    chunks_dir = './chunks'

    # Criar o diretório /chunks se não existir
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    # Carregar o PDF e dividir em páginas usando PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,  # tamanho do chunk
        chunk_overlap=200  # sobreposição entre chunks
    ))

    chunks = []
    chunk_references = []

    # Processar os documentos carregados
    for doc in documents:
        chunk = doc.page_content
        pages = doc.metadata.get('page', 'desconhecida')
        chunk_references.append((pages, 0, len(chunk)))  # Aqui, 0 e len(chunk) são placeholders
        chunks.append(chunk)

    # Carregar o modelo de embeddings local
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Gerar embeddings para os chunks
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Salvar os chunks e os embeddings no diretório /chunks
    for i, (chunk, embedding, reference) in enumerate(zip(chunks, embeddings, chunk_references)):
        chunk_with_reference = f"Pages: {reference[0]}, Start: {reference[1]}, End: {reference[2]}\n{chunk}"
        chunk_txt_path = os.path.join(chunks_dir, f'chunk_{i+1}.txt')
        save_text_to_file(chunk_with_reference, chunk_txt_path)
        print(f'Chunk {i+1} salvo em {chunk_txt_path}')
        
        embedding_txt_path = os.path.join(chunks_dir, f'embedding_{i+1}.txt')
        save_text_to_file(str(embedding), embedding_txt_path)
        print(f'Embedding {i+1} salvo em {embedding_txt_path}')

    # Opcional: salvar o texto completo em um único arquivo
    full_text = "\n".join(chunks)
    save_text_to_file(full_text, txt_path)

if __name__ == '__main__':
    main()
