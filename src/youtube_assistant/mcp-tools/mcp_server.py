from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

# pyrefly: ignore [missing-import]
from fastmcp import FastMCP
from youtube_assistant.rag_pipeline import index_video, retrieve_context, get_full_transcript
from youtube_assistant.rag_pipeline import find_timestamp as _rag_find_timestamp
# pyrefly: ignore [missing-import]
from youtube_assistant.client.youtube_utils import extract_video_id

mcp = FastMCP("YouTube Learning Assistant MCP Server")
_current_video = {"video_id": None}

@mcp.tool()
def load_youtube_video(video_url: str) -> str:
    """
    Fetches the Transcript of a Youtube video and indexes it for
    question-answering. Call this first before answering any question
    about a video's content, if no video has been loaded yet.

    Args: 
        video_url: A Full Youtube URL or bare video Id.
    """

    info =index_video(video_url)
    _current_video["video_id"] = info["video_id"]
    return (
        f"Successfully loaded and indexed video '{info['video_id']}'."
        f"Transcript length : {info['transcript_chars']} Characters,"
        f"split into {info['num_chunks']} chunks"
        f"You can now answer questions about this video's content"
    )

@mcp.tool()
def search_video_content(question: str) -> str:
    """
    Searches the currently loaded video's transcript for content relevant
    to the given question, using vector similarity search. Use this AFTER
    load_youtube_video has been called.

    Args:
        question: The user's question, used as the search query.
    """
    video_id = _current_video["video_id"]
    if video_id is None:
        return "No video has been loaded yet. Ask the user for a Youtube URL and call load_youtube_video first"
    return retrieve_context(video_id, question, top_k = 3)

@mcp.tool()
def get_transcript_overview() -> str:
    video_id = _current_video["video_id"]
    if video_id is None:
        return "No video has been loaded yet. Ask the user for a Youtube UTL and call load_youtube_video first."
    return get_full_transcript(video_id)

@mcp.tool()
def find_timestamp(query: str) -> str:
    video_id = _current_video["video_id"]
    if video_id is None:
        return "No video has been loaded yet. Ask the user for a YouTube URL and call load_youtube_video first"
    return _rag_find_timestamp(video_id, query, top_k = 3)

@mcp.tool()
def get_video_id(video_url: str) -> str:
    """
    Extracts and returns the bare 11-Char youtube video Id from any
    url format. Useful when ypou need to confirm which video is being referred to

    Args:
        video_url : A full Youtube Url or a Bare video Id.
    """
    return extract_video_id(video_url)

if __name__ == "__main__":
    mcp.run()


