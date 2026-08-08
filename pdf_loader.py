import os
from io import BytesIO
from typing import List, Union
from pypdf import PdfReader
from langchain_core.documents import Document
from utils import PDFProcessingError, get_logger

logger = get_logger("pdf_loader")
SUPPORTED_EXTENSION = ".pdf"

def _read_pdf_bytes(file_obj: Union[str, BytesIO], filename: str) -> PdfReader:
    try:
        return PdfReader(file_obj)
    except Exception as exc:  # noqa: BLE001
        raise PDFProcessingError(
            f"'{filename}' could not be read. It may be corrupted, "
            f"password-protected, or not a valid PDF file."
        ) from exc

def load_pdf(file_obj: Union[str, BytesIO], filename: str = None) -> List[Document]:
    if filename is None:
        filename = file_obj if isinstance(file_obj, str) else "uploaded.pdf"
    if not filename.lower().endswith(SUPPORTED_EXTENSION):
        raise PDFProcessingError(
            f"'{filename}' is not a PDF file. Only {SUPPORTED_EXTENSION} files are supported."
        )
    if isinstance(file_obj, str) and not os.path.exists(file_obj):
        raise PDFProcessingError(f"File not found: '{file_obj}'")
    reader = _read_pdf_bytes(file_obj, filename)
    if len(reader.pages) == 0:
        raise PDFProcessingError(f"'{filename}' has no pages.")
    documents: List[Document] = []
    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract page %s of '%s': %s", page_number, filename, exc)
            text = ""
        text = text.strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_number},
                )
            )
    if not documents:
        raise PDFProcessingError(
            f"'{filename}' contains no extractable text. It may be a scanned "
            f"image PDF that requires OCR, which is not supported here."
        )
    logger.info("Loaded %s page(s) of text from '%s'", len(documents), filename)
    return documents

def load_pdfs(files: List[Union[str, BytesIO]], filenames: List[str] = None) -> List[Document]:
    if not files:
        raise PDFProcessingError("No files were provided.")
    if filenames is None:
        filenames = [f if isinstance(f, str) else getattr(f, "name", "uploaded.pdf") for f in files]
    all_documents: List[Document] = []
    errors: List[str] = []
    for file_obj, filename in zip(files, filenames):
        try:
            all_documents.extend(load_pdf(file_obj, filename))
        except PDFProcessingError as exc:
            logger.error(str(exc))
            errors.append(str(exc))
    if not all_documents:
        raise PDFProcessingError(
            "None of the uploaded files could be processed:\n" + "\n".join(errors)
        )
    if errors:
        logger.warning("%d file(s) skipped due to errors: %s", len(errors), errors)
    return all_documents