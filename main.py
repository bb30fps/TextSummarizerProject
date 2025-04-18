import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from transformers import pipeline
import torch
import docx
import pdfplumber
import cv2
import pytesseract
import random
from PIL import Image, ImageTk
import os
import sys
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')


class TextSummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarizer Pro Max")
        master.geometry("1920x1080")

        # Application state
        self.abstractive_model = None
        self.contractions = self._create_contraction_map()
        self.current_image = None
        self.tesseract_path = self._find_tesseract()

        # Initialize UI
        self._setup_interface()
        self._load_models()

        # Default values
        self.length_entry.insert(0, "3")
        self.model_selector.current(0)

    def _find_tesseract(self):
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ]

        for path in paths:
            if os.path.exists(path):
                return path

        messagebox.showwarning("Tesseract Missing",
                               "Tesseract OCR not found. Please install it and set the path in settings.")
        return None

    def _create_contraction_map(self):
        """Dictionary for expanding contractions"""
        return {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                           "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

    def _update_ui(self, event=None):
        """Update UI elements based on selected summarization mode"""
        method = self.model_var.get()
        if method == "abstractive":
            self.length_label.config(text="Max Summary Length (words):")
            self.length_entry.delete(0, tk.END)
            self.length_entry.insert(0, "150")
        else:
            self.length_label.config(text="Summary Length (sentences):")
            self.length_entry.delete(0, tk.END)
            self.length_entry.insert(0, "3")

    def _setup_interface(self):  # GUI Components
        # Main container
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel (Input/Controls)
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel (Output/Preview)
        right_panel = ttk.Frame(main_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(20, 0))

        # Input Section
        input_frame = ttk.LabelFrame(
            left_panel, text="Input Text", padding=(10, 5))
        input_frame.pack(fill=tk.BOTH, expand=True)

        self.input_text = tk.Text(input_frame, wrap=tk.WORD, height=15,
                                  font=('Segoe UI', 10), padx=10, pady=10)
        self.input_scroll = ttk.Scrollbar(
            input_frame, command=self.input_text.yview)
        self.input_text.configure(yscrollcommand=self.input_scroll.set)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Control Panel
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(control_frame, text="Method:").grid(
            row=0, column=0, padx=(0, 5))
        self.model_var = tk.StringVar(value="extractive")
        self.model_selector = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["extractive", "abstractive"],
            state="readonly",
            width=12
        )
        self.model_selector.grid(row=0, column=1, padx=5)

        self.length_label = ttk.Label(
            control_frame, text="Length (sentences):")
        self.length_label.grid(row=0, column=2, padx=(20, 5))
        self.length_entry = ttk.Entry(control_frame, width=8)
        self.length_entry.grid(row=0, column=3, padx=5)

        self.upload_btn = ttk.Button(control_frame, text="📁 Upload File",
                                     command=self._handle_file_upload, style='Accent.TButton')
        self.upload_btn.grid(row=0, column=4, padx=(20, 5))

        self.process_btn = ttk.Button(control_frame, text="⚡ Process",
                                      command=self._generate_summary, style='Accent.TButton')
        self.process_btn.grid(row=0, column=5, padx=5)

        # Output Section
        output_frame = ttk.LabelFrame(
            right_panel, text="Summary", padding=(10, 5))
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=15,
                                   font=('Segoe UI', 10), padx=10, pady=10)
        self.output_scroll = ttk.Scrollbar(
            output_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=self.output_scroll.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Image Preview Section
        preview_frame = ttk.LabelFrame(
            right_panel, text="Image Preview", padding=(10, 5))
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.image_preview = ttk.Label(preview_frame, background='white')
        self.image_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # OCR Controls
        ocr_controls = ttk.Frame(preview_frame)
        ocr_controls.pack(fill=tk.X, pady=(5, 0))

        self.ocr_btn = ttk.Button(ocr_controls, text="📷 Extract from Image",
                                  command=self._process_image, style='Accent.TButton')
        self.ocr_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.ocr_status = ttk.Label(
            ocr_controls, text="Ready", foreground="gray")
        self.ocr_status.pack(side=tk.LEFT)

        # Progress Bar
        self.progress = ttk.Progressbar(
            left_panel, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

        # Bind events
        self.model_selector.bind("<<ComboboxSelected>>", self._update_ui)

    def _arrange_widgets(self):  # Position all UI elements with OCR components

        # Image Preview
        self.image_preview.grid(
            row=0, column=5, rowspan=2, padx=10, pady=10, sticky=tk.NSEW)
        self.ocr_btn.grid(row=2, column=5, padx=10, pady=10)

        # Input area
        self.input_label.grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
        self.input_text.grid(row=1, column=0, columnspan=5,
                             padx=15, pady=5, sticky=tk.NSEW)
        self.input_scroll.grid(row=1, column=5, sticky=tk.NS)

        # Controls
        self.model_selector.grid(
            row=2, column=0, padx=15, pady=12, sticky=tk.W)
        self.length_label.grid(row=2, column=1, padx=10, pady=12)
        self.length_entry.grid(row=2, column=2, padx=10, pady=12)
        self.upload_btn.grid(row=2, column=3, padx=15, pady=12)
        self.summarize_btn.grid(row=2, column=4, padx=15, pady=12)

        # Output
        self.output_label.grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)
        self.output_text.grid(row=4, column=0, columnspan=5,
                              padx=15, pady=5, sticky=tk.NSEW)
        self.output_scroll.grid(row=4, column=5, sticky=tk.NS)

    def _load_models(self):
        try:
            self.abstractive_model = pipeline(
                "summarization",
                model="philschmid/bart-large-cnn-samsum",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            messagebox.showwarning(
                "Model Error", f"Abstractive model failed to load:\n{str(e)}")

    def _enhance_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)

        return sharpened

    def _process_image(self):
        if not self.tesseract_path:
            self._handle_missing_tesseract()
            return

        try:
            self.progress['value'] = 0
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

            img_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if img_path:
                self.ocr_status.config(
                    text="Processing...", foreground="orange")
                self.progress['value'] = 20
                self.master.update()

                # Load and enhance image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Unsupported image format")

                processed_img = self._enhance_image(img)
                self.progress['value'] = 50

                # OCR with advanced config
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(
                    processed_img,
                    config=custom_config,
                    lang='eng'
                )
                self.progress['value'] = 80

                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(tk.END, text)

                # Image preview
                preview_img = Image.fromarray(processed_img)
                preview_img.thumbnail((600, 400))
                self.current_image = ImageTk.PhotoImage(preview_img)
                self.image_preview.config(image=self.current_image)

                self.ocr_status.config(text="Success", foreground="green")
                self.progress['value'] = 100

        except Exception as e:
            self.ocr_status.config(text="Error", foreground="red")
            messagebox.showerror(
                "OCR Error", f"Image processing failed: {str(e)}")
        finally:
            self.progress['value'] = 0

    def _handle_file_upload(self):
        file_types = [
            ("Text Files", "*.txt"),
            ("Word Documents", "*.docx"),
            ("PDF Files", "*.pdf"),
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if file_path:
            try:
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self._process_image(file_path)
                else:
                    content = self._read_file(file_path)
                    self.input_text.delete(1.0, tk.END)
                    self.input_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror(
                    "File Error", f"Failed to read file:\n{str(e)}")

    def _handle_missing_tesseract(self):
        response = messagebox.askyesno(
            "Tesseract Missing",
            "Tesseract OCR not found. Would you like to browse for the executable?"
        )
        if response:
            path = filedialog.askopenfilename(
                title="Select Tesseract Executable")
            if path and os.path.isfile(path):
                self.tesseract_path = path
                pytesseract.pytesseract.tesseract_cmd = path
            else:
                messagebox.showinfo(
                    "Install Tesseract",
                    "Please install Tesseract OCR from:\n"
                    "https://github.com/tesseract-ocr/tesseract"
                )

    def _read_file(self, path):  # Read content from different file formats
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif path.endswith('.docx'):
            doc = docx.Document(path)
            return '\n'.join([p.text for p in doc.paragraphs])
        elif path.endswith('.pdf'):
            with pdfplumber.open(path) as pdf:
                return '\n'.join([page.extract_text() for page in pdf.pages])
        raise ValueError("Unsupported file format")

    def _generate_summary(self):  # Main summary generation
        try:
            input_text = self.input_text.get(1.0, tk.END).strip()
            if not input_text:
                messagebox.showwarning(
                    "Input Error", "Please enter text or upload a file")
                return

            if self.model_var.get() == "extractive":
                summary = self._extractive_summary(input_text)
            else:
                summary = self._abstractive_summary(input_text)

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, summary)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def _extractive_summary(self, text):
        """Improved extractive summarization with intelligent randomization"""
        try:
            num_sentences = int(self.length_entry.get())
            if num_sentences <= 0:
                raise ValueError("Number of sentences must be positive")

            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text

            # Enhanced text processing
            expanded = self._expand_contractions(text)
            words = [
                w.lower() for w in word_tokenize(expanded)
                if w.isalnum() and w not in stopwords.words('english')
            ]

            # Calculate word frequencies with normalization
            freq_dist = FreqDist(words)
            max_freq = max(freq_dist.values()) if freq_dist else 1

            # Score sentences with diversity factor
            scores = []
            for idx, sent in enumerate(sentences):
                sent_words = word_tokenize(
                    self._expand_contractions(sent).lower())
                valid_words = [w for w in sent_words if w in freq_dist]

                if not valid_words:
                    score = 0.0
                else:
                    # Normalized frequency score
                    freq_score = sum(freq_dist[w]
                                     for w in valid_words) / len(valid_words)
                    freq_score /= max_freq  # Normalize to 0-1 range

                    # Positional bias (favor earlier sentences)
                    position_score = 1 - (idx / len(sentences))

                    # Combine scores with randomness
                    score = (0.7 * freq_score) + \
                        (0.2 * position_score) + (0.1 * random.random())

                scores.append((idx, sent, score))

            # Select diverse sentences
            sorted_sentences = sorted(scores, key=lambda x: x[2], reverse=True)
            top_candidates = sorted_sentences[:min(
                num_sentences*3, len(sorted_sentences))]
            selected = random.sample(top_candidates, min(
                num_sentences, len(top_candidates)))
            selected.sort(key=lambda x: x[0])  # Maintain original order

            return ' '.join([s[1] for s in selected])

        except ValueError as e:
            raise ValueError(f"Extractive error: {str(e)}")

    # Generate abstractive summary with precise length control
    def _abstractive_summary(self, text):
        if not self.abstractive_model:
            raise ValueError("Abstractive model not loaded")

        try:
            target_words = int(self.length_entry.get())
            summary = self.abstractive_model(
                text,
                max_length=int(target_words * 2),
                min_length=int(target_words * 0.9),
                do_sample=True,
                temperature=0.85,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.1,
                num_beams=4,
                truncation=True)[0]['summary_text']

            return self._process_abstractive_output(summary, target_words)
        except ValueError:
            raise ValueError("Invalid word limit")

    # Post-process abstractive summary to meet word limit
    def _process_abstractive_output(self, summary, target):
        # Sentence-based trimming
        sentences = sent_tokenize(summary)
        word_count = 0
        selected = []

        for sent in sentences:
            words = word_tokenize(sent)
            if word_count + len(words) > target:
                remaining = target - word_count
                if remaining > 3:
                    selected.append(' '.join(words[:remaining]))
                break
            selected.append(sent)
            word_count += len(words)

        # Fallback truncation if needed
        if word_count < (target * 0.8):
            words = word_tokenize(summary)
            return ' '.join(words[:target]).capitalize()

        # Final cleanup
        processed = ' '.join(selected)
        processed = processed.replace(" .", ".").replace(" ,", ",")
        processed = processed.replace(" 's", "'s").replace(" n't", "n't")
        return processed[0].upper() + processed[1:]

    def _expand_contractions(self, text):  # Expand contractions in text
        words = word_tokenize(text)
        expanded = []
        for word in words:
            lower = word.lower()
            if lower in self.contractions:
                replacement = self.contractions[lower]
                if word.istitle():
                    replacement = replacement.title()
                elif word.isupper():
                    replacement = replacement.upper()
                expanded.append(replacement)
            else:
                expanded.append(word)
        return ' '.join(expanded)


if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()
