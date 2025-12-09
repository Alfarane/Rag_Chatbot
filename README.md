# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that answers questions based on PDF documents using Google Gemini embeddings and FAISS vector search.

## Features

- 📄 **PDF Document Processing**: Load and process PDF documents for knowledge base
- 🔍 **Semantic Search**: Uses FAISS vector index for efficient similarity search
- 🤖 **AI-Powered Answers**: Leverages Google Gemini for embeddings and text generation
- 💬 **Interactive Chat Interface**: Clean, user-friendly Streamlit web interface
- 📚 **Context-Aware Responses**: Retrieves relevant document chunks to provide accurate answers

## Requirements

- Python 3.8+
- Google API Key (for Gemini API)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_chatbot
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install streamlit google-genai faiss-cpu pypdf numpy
```

## Configuration

Set your Google API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

**macOS/Linux:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

To get a Google API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and set it as shown above

## Usage

1. Place your PDF file in the `data/` directory (default: `data/jio_info.pdf`)

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. Start asking questions about your PDF document!

## Project Structure

```
rag_chatbot/
├── app.py                 # Streamlit web application
├── src/
│   ├── build_index.py    # FAISS index builder
│   ├── chunker.py         # Text chunking utilities
│   ├── gemini_client.py   # Google Gemini API client
│   ├── pdf_loader.py      # PDF text extraction
│   ├── rag_answer.py      # RAG answer generation
│   ├── retrieval.py       # Vector similarity search
│   └── main.py            # CLI version (optional)
├── data/
│   └── jio_info.pdf      # Sample PDF document
└── README.md             # This file
```

## How It Works

1. **Document Loading**: PDF is loaded and text is extracted
2. **Chunking**: Text is split into overlapping chunks for better context
3. **Embedding**: Each chunk is embedded using Google Gemini's embedding model
4. **Indexing**: Embeddings are stored in a FAISS vector index
5. **Query Processing**: User questions are embedded and searched against the index
6. **Answer Generation**: Relevant chunks are retrieved and used as context for Gemini to generate answers

## Command Line Interface

The project also includes a CLI version in `src/main.py`:

```bash
cd src
python main.py
```

## Troubleshooting

- **Missing API Key**: Ensure `GOOGLE_API_KEY` is set before running the app
- **PDF Not Found**: Make sure your PDF file exists in the `data/` directory
- **Import Errors**: Ensure all dependencies are installed and you're using the correct Python environment

## License

This project is open source and available for personal and educational use.

