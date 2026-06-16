from langchain_community.vectorstores import FAISS

INDEX_PATH = "faiss_index"

def create_vector_store(chunks, embeddings):

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(INDEX_PATH)

    return vector_store


def load_vector_store(embeddings):

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )