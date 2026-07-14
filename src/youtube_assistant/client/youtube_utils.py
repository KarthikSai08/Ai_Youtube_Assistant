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

def fetch_transcript_text(url_or_id: str) -> str:
    """
    Fetches the transcript for a video and returns it as One Flat string.
    """

    video_id = extract_video_id(url_or_id)
    ytt_api = YouTubeTranscriptApi()

    try:
        fetched = ytt_api.fetch(video_id, languages = ["en", "en-US", "en-GB"])
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

    full_text = " ".join(snippet.text for snippet in fetched)
    return full_text

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Extracted ID :",extract_video_id(test_url))
    try:
        text = fetch_transcript_text(test_url)
        print("Transcript Preview :", text[:300])
    except RuntimeError as e:
        print("Error :",e)