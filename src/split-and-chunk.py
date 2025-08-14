from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from pathlib import Path

# langchain implementation to embed

# create paths to the repo root
repo_root = Path(__file__).resolve().parents[1]
raw_dir = repo_root / "data" / "raw"
emb_dir = repo_root / "data" / "embeddings"

# load from your saved files
loader = DirectoryLoader(
    str(raw_dir), 
    glob="*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)
documents = loader.load()

print(f"Loaded {len(documents)} PyTorch documentation files")

# basic chunking - to improve this try semantic chunking based on function or class (like page i.e. .nn.html, etc.)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", "```"]  # Code-aware splits
)
splits = text_splitter.split_documents(documents)

print(f"Created {len(splits)} chunks from {len(documents)} documents")

# create vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(emb_dir),
    embedding_function=embeddings
)
print(f"Embeddings created and saved to: {emb_dir}")