from rag.loader import load_pdf
from rag.ragchunker import create_chunks
from rag.embedding import get_embeddings
from rag.vector_store import create_vector_store
from rag.vector_store import load_vector_store
from rag.llm import generate_answer


PDF_PATH = "pdf/python.pdf"


def build_index():

    docs = load_pdf(PDF_PATH)

    chunks = create_chunks(docs)

    embeddings = get_embeddings()

    create_vector_store(
        chunks,
        embeddings
    )

    print("Vector DB Created Successfully")


def ask_question(question):

    embeddings = get_embeddings()

    db = load_vector_store(
        embeddings
    )

    docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only from the given context.

Context:
{context}

Question:
{question}
"""

    return generate_answer(prompt)