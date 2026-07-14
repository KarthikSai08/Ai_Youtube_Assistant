from langchain_core.tools import tool
from youtube_assistant.rag_pipeline import index_video, retrieve_context

_current_video = {"video_id" : None}

@tool
def load_youtube_video(video_url : str) -> str:
    """
    Fetches the transcript of youtube video and indexes it for 
    question-answering. Always call this first before answering any 
    question about a video's content, if no video has been loaded 
    yet in this conversation.

    Args : 
        video_url : a full youtube url or a video ID.
    """
    info = index_video(video_url)
    _current_video["video_id"] = info["video_id"]
    return (
        f"Successfully loaded and indexed video '{info['video_id']}."
        f"Transcript length: {info['transcript_chars']} characters,"
        f"split into {info['num_chunks']} chunks"
        f"You can now answer questions about this video's content"
    )

@tool
def search_video_content(question: str) -> str:
    """
    Searches the currently loaded video's transcript for content relavent
    to the given question, usin g vector similarity search, Use this After 
    load_youtube_video has been called, whenever the user asks something about the video's content
    """

    video_id = _current_video["video_id"]
    if video_id is None:
        return "No video has been loaded yet. Ask the uuser for a youtube utl and call load_youtube_video first"
    return retrieve_context(video_id, question, top_k=3)

ALL_TOOLS = [load_youtube_video, search_video_content]    