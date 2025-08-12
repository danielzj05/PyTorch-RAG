# PyTorch-RAG: Friendly PyTorch Documentation Q&A

## Overview
PyTorch-RAG is a Retrieval-Augmented Generation (RAG) system designed to make PyTorch and machine learning documentation more accessible for beginners. It allows you to ask questions about PyTorch and get answers directly from the official documentation using modern ML techniques.

## Features
- Scrapes official PyTorch documentation (neural network modules by default)
- Chunks and embeds docs for efficient retrieval
- Uses a code-focused language model to answer questions
- Fast GPU inference support
- Easy to extend to other PyTorch modules

## How It Works
1. **Scrape the Docs (Run Once!)**
   - The scraper downloads and saves PyTorch documentation locally.
   - **You only need to run the scraper once unless the docs change!**
2. **Chunk and Embed**
   - The text is split into manageable chunks and embedded for semantic search.
3. **Ask Questions**
   - The RAG pipeline retrieves relevant chunks and generates answers using a language model.

## Getting Started
1. Clone the repo and set up your Python environment:
   ```
   git clone <your-repo-url>
   cd pytorch-rag
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Run the scraper (only ONCE):**
   ```
   python src/scraper.py
   ```
3. **Chunk and embed the docs:**
   ```
   python src/split-and-chunk.py
   ```
4. **Ask questions:**
   ```
   python src/rag.py
   ```

## Customization
- To scrape more modules, edit `navigation_pages` in `src/scraper.py`.
- Tune chunk size and overlap in `src/split-and-chunk.py` for best retrieval results.
- Swap out the LLM in `src/rag.py` for different models.

## Notes
- The scraper is designed to run only once. Re-run it only if the documentation updates.
- All data and embeddings are stored locally and excluded from git via `.gitignore`.

## License
MIT

## Contributing
Pull requests and suggestions are welcome!
