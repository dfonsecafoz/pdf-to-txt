import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    text = ''
    references = []

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        start_pos = len(text)
        text += page_text
        end_pos = len(text)
        references.append((page_num + 1, start_pos, end_pos))  # Pages are 1-indexed

    pdf_file.close()
    return text, references

def save_text_to_file(text, txt_path):
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text)

def find_page_for_chunk(chunk_start_pos, chunk_end_pos, references):
    pages = set()
    for page_num, start_pos, end_pos in references:
        if start_pos <= chunk_start_pos < end_pos or start_pos < chunk_end_pos <= end_pos:
            pages.add(page_num)
    return pages

def main():
    pdf_path = 'meucodigo.pdf'
    txt_path = 'meucodigo.txt'
    chunks_dir = './chunks'

    # Criar o diretório /chunks se não existir
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    text, references = extract_text_from_pdf(pdf_path)
    print(text)

    # Utilizar RecursiveCharacterTextSplitter para dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # tamanho do chunk
        chunk_overlap=200,  # sobreposição entre chunks
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    chunk_references = []

    current_pos = 0
    for chunk in chunks:
        chunk_start_pos = text.find(chunk, current_pos)
        chunk_end_pos = chunk_start_pos + len(chunk)
        current_pos = chunk_end_pos - 200  # Ajustar a posição atual para considerar a sobreposição

        # Encontrar as referências de página correspondentes ao chunk usando a função find_page_for_chunk
        pages = find_page_for_chunk(chunk_start_pos, chunk_end_pos, references)
        page_str = ', '.join(map(str, pages))
        chunk_references.append((page_str, chunk_start_pos, chunk_end_pos))

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
    save_text_to_file(text, txt_path)

if __name__ == '__main__':
    main()
