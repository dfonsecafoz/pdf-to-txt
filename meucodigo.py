import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    text = ''
    
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    pdf_file.close()
    return text

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

    text = extract_text_from_pdf(pdf_path)
    print(text)

    # Utilizar RecursiveCharacterTextSplitter para dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # tamanho do chunk
        chunk_overlap=200,  # sobreposição entre chunks
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    # Carregar o modelo de embeddings local
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Gerar embeddings para os chunks
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Salvar os chunks e os embeddings no diretório /chunks
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_txt_path = os.path.join(chunks_dir, f'chunk_{i+1}.txt')
        save_text_to_file(chunk, chunk_txt_path)
        print(f'Chunk {i+1} salvo em {chunk_txt_path}')
        
        embedding_txt_path = os.path.join(chunks_dir, f'embedding_{i+1}.txt')
        save_text_to_file(str(embedding), embedding_txt_path)
        print(f'Embedding {i+1} salvo em {embedding_txt_path}')

    # Opcional: salvar o texto completo em um único arquivo
    save_text_to_file(text, txt_path)

if __name__ == '__main__':
    main()
