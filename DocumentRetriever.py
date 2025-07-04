import os
import numpy as np
import faiss
import torch
import pdfplumber
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.CRITICAL)
# Suppress INFO and WARNING logs from pdfplumber
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
import re
# from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DocumentRetriever:
    def __init__(self, data_dir='Data', chunk_size=500):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.pdf_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdf')]
        self.embeddings_model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.embeddings = SentenceTransformer(self.embeddings_model,device=device)
        self.documents = []
        self.page_to_chunk_mapping = []
        self.index = None
        self.build_index()
    
    def load_documents(self, path):
        documents = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    lines = [line for line in lines if not (
                        line.strip().lower().startswith('doi') or 
                        'copyright' in line.lower() or 
                        'http' in line.lower() or 
                        'figure' in line.lower() or
                        line.strip().isdigit()
                        )]
                    text = ' '.join(lines)
                    text = re.sub(r'\[\s?\d+(?:\s?,\s?\d+)*\s?\]', '', text)
                    text = re.sub(r'\(\s?\d+(?:\s?,\s?\d+)*\s?\)', '', text)
                    text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\.?\s*)?(?:[A-Za-z\s]+\.)?\s?\d{4};\d+\(.*?\):\d+(–\d+)?\.', '', text)
                    sentences = sent_tokenize(text)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk.split()) + len(sentence.split()) <= self.chunk_size:
                            current_chunk += " " + sentence
                        else:
                            # Otherwise, finalize the current chunk and start a new one
                            if current_chunk:
                                documents.append({
                                    "chunk": len(documents),
                                    "text": current_chunk.strip(),
                                    "pages": [page_num],
                                    "file": path
                                })
                            current_chunk = sentence
                    # Add the last remaining chunk
                    if current_chunk:
                         documents.append({
                            "chunk": len(documents),
                            "text": current_chunk.strip(),
                            "pages": [page_num],
                            "file": path
                        })
        return documents

    def build_index(self):
        # if os.path.exists('faiss_index_commonly_based.idx'):
            # print(f"Loading FAISS index...")
            # self.index = faiss.read_index('faiss_index_commonly_based.idx')
            # return
        # print("Building FAISS index from scratch...")
        for pdf_path in self.pdf_paths:
            self.documents.extend(self.load_documents(pdf_path))
        cleaned_texts = [doc['text'].strip() for doc in self.documents if isinstance(doc.get('text'), str) and doc['text'].strip()]
        embeddings = self.embeddings.encode(cleaned_texts, convert_to_tensor=False,show_progress_bar=True,batch_size=64)
        embeddings_np = np.array(embeddings)
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        # faiss.write_index(self.index, 'faiss_index_commonly_based.idx')
        # print("FAISS index saved")

    def query(self, query_text, k=5):
        query_embedding = self.embeddings.encode([query_text], convert_to_tensor=False, show_progress_bar=True,batch_size=64)
        query_embedding_np = np.array(query_embedding)
        distances, indices = self.index.search(query_embedding_np, k)
        results = [self.format_result(i, distances[0][j]) for j, i in enumerate(indices[0])]
        return sorted(results, key=lambda x: x['distance'])
    
    def format_result(self, index, distance):
        doc = self.documents[index]
        return {
            "text": doc["text"],
            "file": doc["file"],
            "pages": doc["pages"],
            "distance": float(distance)
        }