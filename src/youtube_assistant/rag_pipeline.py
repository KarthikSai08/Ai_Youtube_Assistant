
# pyrefly: ignore [missing-import]
from pydantic_core.core_schema import NoInfoValidatorFunctionSchema
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from youtube_assistant.settings import VECTORSTORE_DIR, EMBEDDING_MODEL_NAME
from youtube_assistant.client.youtube_utils import extract_video_id, fetch_transcript_segments, format_timestamp

embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)

_transcript_cache : dict[str, str] = {}

def _qa_collection_name(video_id: str) -> str:
    """
    QA Collections : chunks of flattened text, no timing info.
    """
    return f"yt_{video_id}"
def _timestamp_collection_name(video_id: str) -> str:
    """
    Separate Collections: chunks of Raw segments, each
    carrying its start-time as metadata. Kept apart from Q&A
    collection on purpose - see module docstring.
    """
    return f"ys_ts_{video_id}"

def _group_segments_into_timestamp_chunks(segments : list[dict], max_chars: int = 300) ->list[dict]:
    chunks = []
    current_text_parts = []
    current_start = None

    for seg in segments :
        if current_start is None:
            current_start = seg["start"]
    current_text_parts.append(seg["text"])
    current_length = sum(len(t) for t  in current_text_parts)
    if current_length >= max_chars:
        chunks.append({"text": " ".join(current_text_parts), "start": current_start})
        current_text_parts = []
        current_start = None

    if current_text_parts:
        chunks.append({"text": " ".join(current_text_parts), "start": current_start})

    return chunks

def index_video(url_or_id: str) -> dict:
    """
    Runs the full indexing pipeline for a video:
      1. fetch RAW timestamped segments 
      2. build the flat transcript from them (for Q&A chunking + cache)
      3. index the Q&A collection (flat-text chunks, as in Part 1-3)
      4. index the NEW timestamp collection (timestamped chunks)

    Returns metadata about what was indexed.
    """
    video_id = extract_video_id(url_or_id)
    segments = fetch_transcript_segments(video_id)
    transcript = fetch_transcript_segments(video_id)
    _transcript_cache[video_id] = transcript

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators= ["\n\n", "\n", ". "," ",""]
    ) 
    qa_chunks = splitter.split_text(transcript)

    qa_collection = _qa_collection_name(video_id)
    Chroma.from_texts(
        texts = qa_chunks,
        embedding = embedding_model,
        collection_name = qa_collection,
        persist_directory = VECTORSTORE_DIR,
        metadatas = [{"video_id": video_id, "chunk_index": i} for i in range(len(qa_chunks))]
    )
    ts_chunks = _group_segments_into_timestamp_chunks(segments)
    if ts_chunks :
        Chroma.from_texts(
            texts=[c["text"] for c in ts_chunks],
            embedding = embedding_model,
            collection_name = _timestamp_collection_name(video_id),
            persist_directory =VECTORSTORE_DIR,
            metadatas = [{"video_id": video_id, "start_seconds" : c["start"]} for c in ts_chunks]
        )

    return {
        "video_id": video_id,
        "num_chunks" : len(qa_chunks),
        "num_timestamp_chunks": len(ts_chunks),
        "transcript_chars" : len(transcript)
    }


def retrieve_context(video_id : str, query: str, top_k : int = 3) -> str:
    collection = _qa_collection_name(video_id)
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

def get_full_transcript(video_id : str, max_chars: int = 8000) -> str:
    transcript = _transcript_cache.get(video_id)
    if transcript is None:
        return "(no transcript cache for this video - has it been loaded yet?)"
    if len(transcript) <= max_chars:
        return transcript
    return transcript[:max_chars] + "\n\n[...transcipt truncated for length ...]"

def find_timestamp(video_id : str, query: str, top_k : int = 3) -> str:
    collection = _timestamp_collection_name(video_id)
    vectorstore = Chroma(
        collection_name = collection,
        embedding_function=embedding_model,
        persist_directory = VECTORSTORE_DIR
    )
    results = vectorstore.similarity_search(query, k = top_k)
    if not results:
        return "(no matching moment found - has this video been indexed yet?)"
    lines = []
    for doc in results :
        start_seconds = doc.metadata.get("start_seconds", 0)
        timestamp = format_timestamp(start_seconds)
        lines.append(f"[{timestamp}] {doc.page_content}")
    return "\n\n".join(lines)
 
if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    info = index_video(test_url)
    print("Indexed :", info)

    ctx = retrieve_context(info["video_id"], "What is this video about")
    print("Retrived context :\n", ctx)

    overview = get_full_transcript(info["video_id"], max_chars = 500)
    print("\nFull-transcript overview:\n", overview)

    timestamp = find_timestamp(info["video_id"], "opening moments")
    print("\nTimestamp Search:\n", timestamp)
    