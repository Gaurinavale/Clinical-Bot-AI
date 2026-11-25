
import pymupdf
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = r"C:\Users\Dell\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin"


# Extract text from the PDF
def load_pdf_file(data):
    pdf = pymupdf.open(data + "Medical_book.pdf")
    documents = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        # Use OCR only when the page has no extractable text
        if not text.strip():
            images = convert_from_path(
                data + "Medical_book.pdf",
                first_page=page_number + 1,
                last_page=page_number + 1,
                poppler_path=POPPLER_PATH
            )
            text = pytesseract.image_to_string(images[0])

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number + 1}
                )
            )

    pdf.close()
    return documents


# Split extracted text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks


# Download Hugging Face embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
