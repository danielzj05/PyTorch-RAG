
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# langchain implementation to embed

# Load from your saved files
loader = DirectoryLoader(
    "../data/raw/", 
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
print("Embeddings created and saved to ../data/embeddings")