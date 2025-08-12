from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import torch

# Load the existing vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="../data/embeddings",
    embedding_function=embeddings
)

print(torch.cuda.is_available())

llm_pipeline = pipeline(
    "text-generation",
    model="Salesforce/codegen-350M-multi",  # Better for code
    device=0,  # Use GPU
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    pad_token_id=50256,
    eos_token_id=50256
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

template = """Use the following pieces of PyTorch documentation to answer the question about PyTorch programming.

Context from PyTorch docs:
{context}

Question: {question}

Please provide a clear, concise answer. If possible, include a code example and reference the relevant PyTorch documentation. If the answer is not in the context, say 'Not found in provided documentation.'
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

if __name__ == "__main__":
    query = "How is torch.nn.Linear implemented?"
    print(f"Question: {query}")

    # Retrieve and filter docs
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    filtered_docs = [doc for doc in retrieved_docs if "torch.nn.Linear" in doc.page_content]
    context = "\n\n".join([doc.page_content for doc in filtered_docs])

    # Run LLM directly with filtered context and question
    prompt_str = prompt.format(context=context, question=query)
    result = llm.invoke(prompt_str)
    print(f"Answer: {result}")