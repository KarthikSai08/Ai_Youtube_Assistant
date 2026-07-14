
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from youtube_assistant.config import VECTORSTORE_DIR, EMBEDDING_MODEL_NAME
from youtube_assistant.client.youtube_utils import extract_video_id, fetch_transcript_text

embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)

def _collection_name(video_id: str) -> str:
    """
    Why : Chroma stores vectors i named "collections". If we dumped every
    video's chunks into ONE collection, a question about Video A could 
    retrieve chunks from video B (topic bleed). giving each video its own 
    collection, keyed by video_id, keeps retrievel scoped correctly."""
    return f"yt_{video_id}"

def index_video(url_or_id: str) -> dict:
    """
    Runs steps 1-4 of the RAG pipeline for a Single Video:
    fetch transcript -> chunk -> embed -> store in chorma.
    
    REturns metadata about what was indexed (used by agent to confirm
    success back to user)
    """
    video_id = extract_video_id(url_or_id)
    transcript = fetch_transcript_text(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators= ["\n\n", "\n", ". "," ",""]
    ) 
    chunks = splitter.split_text(transcript)

    collection = _collection_name(video_id)
    vectorstore = Chroma.from_texts(
        texts = chunks,
        embedding = embedding_model,
        collection_name = collection,
        persist_directory = VECTORSTORE_DIR,
        metadatas = [{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))]
    )

    return {
        "video_id": video_id,
        "num_chunks" : len(chunks),
        "transcript_chars" : len(transcript)
    }


def retrieve_context(video_id : str, query: str, top_k : int = 3) -> str:
    """
    Runs setp 5-9 of the RAG pipeline: embed the query, simlarity search
    agains the video's collection, returns top_k chunks joined into a
    single context string ready to inject into a prompt.
    """

    collection = _collection_name(video_id)
    vectorstore = Chroma(
        collection_name = collection,
        embedding_function=embedding_model,
        persist_directory = VECTORSTORE_DIR
    )

    results = vectorstore.similarity_search(query, k = top_k)

    if not results:
        return "(No relavant context found!!! - Check wheather the video has been indexed)"

    context = "\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(results)
    )
    return context

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    info = index_video(test_url)
    print("Indexed :", info)
    ctx = retrieve_context(info["video_id"], "What is this video about")
    print("Retrived context :\n", ctx)