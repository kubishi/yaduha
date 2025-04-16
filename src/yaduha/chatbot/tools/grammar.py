import argparse
import base64
import logging
from typing import List
from pdf2image import convert_from_path, pdfinfo_from_path
from openai import OpenAI
import os
from dotenv import load_dotenv
import pathlib
import pandas as pd
from pinecone import Pinecone

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc_index = pc_client.Index(os.getenv("PINECONE_INDEX"))

def create_embeddings(text_dir: pathlib.Path, chunk_size: int = 500) -> None:
    '''
        Generates embeddings from the paragraph txt file.
        Stores these embeddings into pinecone database
        Params:
            pdf: str =  file name

        returns:
            Nothing, but adds the data into pinecone database
    '''
    rows = []
    for page in text_dir.glob("*.txt"):
        text = page.read_text()
        for idx, paragraph in enumerate(text.split('\n\n')):
            page_num = int(page.stem.split('_')[-1])
            rows.append((text_dir.name, page_num, idx, paragraph))
    df = pd.DataFrame(rows, columns=['file', 'page', 'paragraph', 'text'])
    
    # remove empty paragraphs
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)
    
    docs = df['text'].tolist()
    doc_chunks = [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
    embeddings = []
    for doc_chunk in doc_chunks:
        res = openai_client.embeddings.create(input=doc_chunk, model=EMBEDDINGS_MODEL)
        embeddings.extend([d.embedding for d in res.data])
    
    vectors = []
    for idx, row in df.iterrows():
        unique_id = f"{row.file}_{row.page}_{row.paragraph}"
        vector = {
            "id": unique_id,
            "values": embeddings[idx],
            "metadata": {
                "text": row.text,
                "file": row.file,
                "page": row.page,
                "paragraph": row.paragraph
            }
        }
        vectors.append(vector)
        
    vector_chunk_size = 100
    vector_chunks = [vectors[i:i+vector_chunk_size] for i in range(0, len(vectors), vector_chunk_size)]
    for chunk in vector_chunks:
        res = pc_index.upsert(vectors=chunk)
    logging.info(res)

def clear_embeddings(image_dir: pathlib.Path) -> None:
    '''
        Clears the embeddings from the pinecone database with file name
    '''
    file_name = image_dir.name
    # delete embeddings where metadata.file == file_name
    res = pc_index.delete(filter=f"metadata.file == '{file_name}'")
    logging.info(res)

def search_grammar(query: str, limit: int = 5) -> list:
    """Searches the grammar pdf for relevant information based on  
    
    Args:
        query (str): The query to search for in the grammar pdf

    Returns:
        list: A list of the top 5 most relevant information based on the query
    """
    q_embeddings = openai_client.embeddings.create(input=query, model=EMBEDDINGS_MODEL).data[0].embedding
    query_result = pc_index.query(vector=q_embeddings, top_k=limit, include_metadata=True)
    relevant_information = [match["metadata"]["text"] for match in query_result["matches"]]
    return relevant_information

def convert_pdf_to_images(pdf_path: pathlib.Path, output_folder: pathlib.Path) -> None:
    """
    Convert a PDF file to a series of images, saving them in the output folder.

    Args:
        pdf_path (pathlib.Path): Path to the PDF file.
        output_folder (pathlib.Path): Path to the directory where images will be saved.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get total number of pages in the PDF
    pdf_info = pdfinfo_from_path(str(pdf_path))
    total_pages = pdf_info["Pages"]

    # Convert pages one by one
    for i in range(1, total_pages + 1):
        logging.info(f"Converting page {i}/{total_pages} of {pdf_path} to image.")
        images = convert_from_path(str(pdf_path), first_page=i, last_page=i)
        if images:
            image_path = output_folder / f"page_{i}.png"
            images[0].save(image_path, "PNG")

def openai_image_to_text(image_path):
    image_data = base64.b64encode(image_path.read_bytes()).decode('utf-8')

    #Image to text using openai
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Use image to text to rewrite exactly what the image says. "
                            "When you see a picture in the text, clearly describe the picture. "
                            "If the page is empty, return an empty string."
                        ), 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
    )

    return response.choices[0].message.content

def ptf_to_text(image_dir: pathlib.Path, text_dir: pathlib.Path) -> None:
    images = sorted(image_dir.glob("*.png"), key=lambda x: int(x.stem.split('_')[-1]))
    text_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        text = openai_image_to_text(image_path)
        text_path = text_dir / f"{image_path.stem}.txt"
        text_path.write_text(text)
        logging.info(f"Converted {image_path} to text.")

def get_parser():
    parser = argparse.ArgumentParser(description="Assistant for Owens Valley Paiute")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare the grammar data, convert pdf to images and text, and create embeddings")
    search_parser = subparsers.add_parser("search", help="Search the grammar data")
    search_parser.add_argument("query", help="The search query")
    search_parser.add_argument("--limit", default=5, type=int, help="The number of results to return")

    return parser

def prepare():
    input_path = thisdir / "input"

    for input_file in input_path.glob("*.pdf"):
        image_dir = thisdir / "page_images" / input_file.stem
        text_dir = thisdir / "page_text" / input_file.stem
        
        if not image_dir.exists():
            convert_pdf_to_images(input_file, image_dir)
        else:
            logging.info(f"{image_dir} already exists. Skipping PDF to image conversion.")

        if not text_dir.exists():
            ptf_to_text(image_dir, text_dir)
        else:
            logging.info(f"{text_dir} already exists. Skipping Image to text conversion.")

        logging.info(f"Creating embeddings for {input_file.stem}")
        # clear_embeddings(image_dir)
        create_embeddings(text_dir)

def search(query, limit):
    results = search_grammar(query, limit)
    print("\n".join(results))

def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    if args.command == "prepare":
        prepare()
    elif args.command == "search":
        search(args.query, args.limit)
    else:
        parser.print_help()
        

if __name__ == "__main__":
    main()