# Text Summarizer Pro Max

**An intelligent summarization toolkit** with multi-format support and AI-powered capabilities

## Features

**Input Options**
   - ✍️ **Direct Input**: Type/paste text into the main text area
   - 📤 **File Upload**: Click "Upload File" button supported formats:
     - Documents: `.txt`, `.docx`, `.pdf`
     - Images: `.jpg`, `.jpeg`, `.png`, `.bmp`

**Mode Selection**
   - 🔍 **Extractive Mode**:
     - Use numeric entry widget to choose sentence count (3-15)
     - Ideal for quick overviews of large documents
   - 🧪 **Abstractive Mode**:
     - Set word limit (150-500 words recommended)
     - First use downloads `bart-large-cnn` model (~1.5GB)
     - Requires stable internet connection

**Generate Summary**
   - Click the "⚡ Process" button
   - Monitor progress via status bar:
     - 📊 Processing stages shown in real-time
     - ⬇️ Model download progress (first abstractive use)
   - Typical processing times:
     - Extractive: 2-5 seconds
     - Abstractive: 10-30 seconds (GPU), 1-2 minutes (CPU)


**Output Handling**
   - 📋 Copy summary to clipboard with right-click menu
   - 💾 Save using "Export" options:
     - Text file (.txt)
     - Word document (.docx)
     - Markdown (.md)
   - 🖼️ For image inputs: 
     - Preview enhanced image in right panel
     - Compare original vs OCR-processed text


**Troubleshooting Common Issues**:
```markdown
- ❗ OCR Failure: Check Tesseract installation path in settings
- 🐢 Slow Performance: Reduce summary length or switch to extractive mode
- 💾 Storage Full: Clear model cache from `~/.cache/huggingface`
- 🔗 Connection Issues: Manual model download available [here](https://huggingface.co/models)


### Core Components
| Component | Technology |
|-----------|------------|
| **NLP Engine** | NLTK, spaCy, Transformers |
| **AI Models** | BART, T5 (Hugging Face) |
| **OCR** | Tesseract 5.0 + OpenCV |
| **GUI** | Custom Tkinter Framework |
| **Processing** | Multi-threaded Architecture |


## Installation

### Setup
```bash
# Clone repository
git clone https://github.com/bb30fps/TextSuummarizerProject.git
cd TextSummarizerProject

# Create virtual environment (recommended)
python -m venv venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords
```

### GUI Workflow
1. **Input Selection**
   - Paste text or choose file (📁 button)
   - Adjust processing parameters
2. **Processing**
   - Real-time progress tracking
   - Automatic error recovery
3. **Output Management**
   - Copy to clipboard
   - Export as Markdown/PDF
   - Share via integrated options


## Usage

1. **Launch Application**
   ```bash
   python main.py
   ```

2. **Input Options**
   - ✍️ Direct text input in the text area
   - 📤 Upload files using the "Upload File" button

3. **Mode Selection**
   - 🔍 **Extractive**: Select number of sentences
   - 🧪 **Abstractive**: Set word limit (150-500 words recommended)

4. **Generate Summary**
   - Click "Generate Summary" button
   - First abstractive use will download AI models (~1.5GB)
     

**Install**:
```bash
# If facing Tkinter issues
python -m pip install python-tk
```

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- Hugging Face for [Transformers](https://huggingface.co/) library
- NLTK team for natural language tools
- Python community for amazing open-source libraries
- Tesseract OCR Community


**Note**: First abstractive summarization may take 2-5 minutes for model download. Subsequent uses will be faster!
```

For best results:
1. Replace placeholder images with actual screenshots
2. Update repository URLs
3. Add actual license file
4. Include system requirements section if needed
5. Add contact information for support





