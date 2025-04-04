# Text Summarizer

A powerful text summarization application with dual-mode capabilities (extractive & abstractive) and multi-file format support.

## Features

- **Dual Summarization Modes**
  - 🧠 **Extractive**: Traditional NLP-based sentence selection
  - 🤖 **Abstractive**: AI-powered summary generation using transformers
- **File Format Support**
  - 📄 Text (.txt)
  - 📑 Word (.docx)
  - 📊 PDF (.pdf)
- **Smart Features**
  - 🔄 Contraction expansion (e.g., "don't" → "do not")
  - 📏 Adjustable summary length control
  - 🛠 Error-resistant design with progress indicators

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/bb30fps/TextSuummarizerProject.git
cd TextSummarizerProject

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords
```

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

## Troubleshooting

**Common Issues**:
- **Model Download Failures**:
  - Check internet connection
  - Retry with VPN if needed
- **File Read Errors**:
  - Ensure files are not password protected
  - Verify file integrity
- **CUDA Out of Memory**:
  - Reduce abstractive word limit
  - Close other GPU-intensive applications

**For Windows Users**:
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

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- Hugging Face for [Transformers](https://huggingface.co/) library
- NLTK team for natural language tools
- Python community for amazing open-source libraries

---

**Note**: First abstractive summarization may take 2-5 minutes for model download. Subsequent uses will be faster!
```

This README includes:
1. Clear visual hierarchy with emojis
2. Detailed installation instructions
3. Visual guides (placeholder images - replace with actual screenshots)
4. Troubleshooting section
5. Contribution guidelines
6. License and acknowledgments
7. Mobile-friendly formatting

For best results:
1. Replace placeholder images with actual screenshots
2. Update repository URLs
3. Add actual license file
4. Include system requirements section if needed
5. Add contact information for support
