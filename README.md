# YouTube Learning Assistant

An AI-powered assistant that lets you **chat with YouTube videos**. Paste a YouTube URL, and the assistant fetches the transcript, indexes it into a vector store, and answers your questions using RAG (Retrieval-Augmented Generation) over the transcript content.

Built with **LangGraph**, **MCP (Model Context Protocol)**, **ChromaDB**, and **Groq**.

---

## How It Works

```
User pastes YouTube URL
        │
        ▼
┌─────────────────────┐
│  Transcript Fetcher  │  youtube-transcript-api
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Text Splitter      │  RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Embeddings         │  sentence-transformers/all-MiniLM-L6-v2
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   ChromaDB           │  One collection per video (avoids topic bleed)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Agent (LangGraph)  │  Calls tools via MCP → retrieves relevant chunks → LLM answers
└─────────┬───────────┘
          ▼
    Answer to user
```

**Agent tools (served via MCP):**

| Tool | Purpose |
|------|---------|
| `load_youtube_video` | Fetches transcript, chunks it, embeds & stores in ChromaDB |
| `search_video_content` | Vector similarity search over the indexed transcript |
| `get_video_id` | Extracts a bare 11-char video ID from any YouTube URL |

---

## Project Structure

```
Ai_Youtube_Assistant/
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   └── youtube_assistant/
│       ├── __init__.py
│       ├── main.py                 # Entry point — interactive CLI chat loop
│       ├── config.py               # Loads .env, creates Groq LLM instance
│       ├── rag_pipeline.py         # RAG: indexing + retrieval logic
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   └── agent_graph.py      # LangGraph agent: builds state graph with MCP tools
│       │
│       ├── client/
│       │   ├── __init__.py
│       │   └── youtube_utils.py    # YouTube transcript fetching & video ID extraction
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   └── mcp_server.py       # MCP server exposing tools over stdio
│       │
│       └── tools/
│           ├── __init__.py
│           └── youtube_tools.py    # Standalone LangChain tools (kept for reference)
│
├── tests/
│   └── __init__.py
│
└── vectorstore/                    # ChromaDB persisted data (auto-created, gitignored)
```

---

## Prerequisites

- **Python 3.10+**
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Ai_Youtube_Assistant.git
cd Ai_Youtube_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |

---

## Running the Program

```bash
python -m youtube_assistant.main
```

### Usage

```
You: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Assistant: Successfully loaded and indexed video 'dQw4w9WgXcQ'. Transcript length: 3420 characters, split into 5 chunks. You can now answer questions about this video's content.

You: What is this video about?
Assistant: This video is about...

You: exit
```

1. **Paste a YouTube URL** to load the video transcript
2. **Ask questions** about the video content
3. Type `exit` or `quit` to stop

---

## Running Tests

Each module has a standalone test at the bottom. Run them individually:

```bash
# Test YouTube transcript fetching
python -m youtube_assistant.client.youtube_utils

# Test RAG pipeline (indexes + retrieves)
python -m youtube_assistant.rag_pipeline

# Test the full agent graph (asks a sample question)
python -m youtube_assistant.agent.agent_graph
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | [Groq](https://groq.com/) via `langchain-groq` |
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Tool serving | [MCP](https://modelcontextprotocol.io/) via `langchain-mcp-adapters` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | [ChromaDB](https://www.trychroma.com/) via `langchain-chroma` |
| Transcript fetching | `youtube-transcript-api` |
| CLI formatting | `rich` |
