import re
# pyrefly: ignore [missing-import]
from youtube_transcript_api import YouTubeTranscriptApi
# pyrefly: ignore [missing-import]
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

def extract_video_id(url_or_id : str) -> str:
    """
    Accept a full Youtube URL (watch?v=, youtu.be/, embed/) OR a bare
    11-Char video ID, and return just the video ID
    """

    url_or_id = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/)([A-Za-z0-9_-]{11})(?:&|\?|$|\/)",
        r"youtu\.be\/([A-Za-z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a video Id From : {url_or_id}")

def _fetch_raw_transcript(video_id : id):
    """
    Shared fetch + error-handling logic used by both
    fetch_transcript_text() and fetch_transcript_segments()
    returns the raw FetchedTranscript object.
    """    
    ytt_api = YouTubeTranscriptApi()

    try:
        return ytt_api.fetch(video_id, languages = ["en", "en-US", "en-GB"])
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcriptions are disabled for video {video_id}")
    except NoTranscriptFound:
        raise RuntimeError(f"No English Transcriptions are found for video  {video_id}")
    except VideoUnavailable:
        raise RuntimeError(f"Video {video_id} is unavailable or private")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch transcript for {video_id}: {e}"
            "If this is a 403/429, YouTube may be rate-limiting your IP"
            "wait for a bit and try , or try a different network"
        )

def fetch_transcript_text(url_or_id: str) -> str:
    """
    Fetches the transcript for a video and returns it as One Flat string.
    """

    video_id = extract_video_id(url_or_id)
    fetched = _fetch_raw_transcript(video_id)
    full_text = " ".join(snippet.text for snippet in fetched)
    return full_text

def fetch_transcript_segments(url_or_id : str) -> list[dict]:
    """
    Fetches the Trancript Withpout Flattening timestamps away
    return a list of {"text" : str, "start" : float} dicts, one
    per captions snippet, in order. This is what powers timestamp search:
    "when does the video mention X" needs to know WHERE in the video a
    piece of text came from, which fetch_transcript_text() throws away
    on purpose (it's the wrong tool for this job - see rag_pipeline.py'
    docstring for thr same "match retrieval strategy to the task"
    lesson applied here)
    """
    video_id = extract_video_id(url_or_id)
    fetched = _fetch_raw_transcript(video_id)
    return [{"text" : snippet.text, "start" : snippet.start} for snippet in fetched]

def format_timestamp(seconds: float) -> str:
    """
    Formats a second count as MM:SS or HH:MM:SS 
    """
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds,3600)
    minutes, secs = dict(remainder, 60)
    if hours: 
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}" 

if __name__ == "__main__":

    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Extracted ID :",extract_video_id(test_url))
    try:
        text = fetch_transcript_text(test_url)
        print("Transcript Preview :", text[:300])
    except RuntimeError as e:
        print("Error :",e)