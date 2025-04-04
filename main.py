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
import random

nltk.download('punkt')
nltk.download('stopwords')


class TextSummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Smart Text Summarizer")
        master.geometry("1100x750")

        # Application state
        self.abstractive_model = None
        self.contractions = self._create_contraction_map()

        # Initialize UI
        self._setup_interface()
        self._load_models()

        # Default values
        self.length_entry.insert(0, "3")
        self.model_selector.current(0)

    def _create_contraction_map(self):
        return {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

    def _setup_interface(self):
        # Configure grid layout
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(4, weight=1)

        # Input section
        self.input_label = ttk.Label(
            self.master, text="Input Text:", font=('Arial', 12))
        self.input_text = tk.Text(self.master, wrap=tk.WORD, height=15,
                                  font=('Arial', 10), padx=10, pady=10)
        self.input_scroll = ttk.Scrollbar(self.master, orient=tk.VERTICAL,
                                          command=self.input_text.yview)
        self.input_text.configure(yscrollcommand=self.input_scroll.set)

        # Control panel
        self.model_var = tk.StringVar(value="extractive")
        self.model_selector = ttk.Combobox(
            self.master,
            textvariable=self.model_var,
            values=["extractive", "abstractive"],
            state="readonly",
            width=12,
            font=('Arial', 11)
        )
        self.length_label = ttk.Label(self.master, text="Summary Length (sentences):",
                                      font=('Arial', 11))
        self.length_entry = ttk.Entry(self.master, width=8, font=('Arial', 11))
        self.upload_btn = ttk.Button(self.master, text="Upload File",
                                     command=self._handle_file_upload)
        self.summarize_btn = ttk.Button(self.master, text="Generate Summary",
                                        command=self._generate_summary,
                                        style='Accent.TButton')

        # Output section
        self.output_label = ttk.Label(self.master, text="Summary:",
                                      font=('Arial', 12))
        self.output_text = tk.Text(self.master, wrap=tk.WORD, height=15,
                                   font=('Arial', 10), padx=10, pady=10)
        self.output_scroll = ttk.Scrollbar(self.master, orient=tk.VERTICAL,
                                           command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=self.output_scroll.set)

        # Arrange components
        self._arrange_widgets()
        self.model_selector.bind("<<ComboboxSelected>>", self._update_ui)

    def _arrange_widgets(self):        #Position all UI elements
        # Input area
        self.input_label.grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
        self.input_text.grid(row=1, column=0, columnspan=5, padx=15, pady=5,
                             sticky=tk.NSEW)
        self.input_scroll.grid(row=1, column=5, sticky=tk.NS)

        # Controls
        self.model_selector.grid(
            row=2, column=0, padx=15, pady=12, sticky=tk.W)
        self.length_label.grid(row=2, column=1, padx=10, pady=12)
        self.length_entry.grid(row=2, column=2, padx=10, pady=12)
        self.upload_btn.grid(row=2, column=3, padx=15, pady=12)
        self.summarize_btn.grid(row=2, column=4, padx=15, pady=12)

        # Output area
        self.output_label.grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)
        self.output_text.grid(row=4, column=0, columnspan=5, padx=15, pady=5,
                              sticky=tk.NSEW)
        self.output_scroll.grid(row=4, column=5, sticky=tk.NS)

    def _load_models(self):       #Initialize abstractive summarization model
        try:
            self.abstractive_model = pipeline(
                "summarization",
                model="philschmid/bart-large-cnn-samsum",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            messagebox.showwarning("Model Error",
                                   f"Abstractive model failed to load:\n{str(e)}")

    def _update_ui(self, event=None):        #Update UI based on selected mode

        mode = self.model_var.get()
        if mode == "abstractive":
            self.length_label.config(text="Max Summary Length (words):")
            self.length_entry.delete(0, tk.END)
            self.length_entry.insert(0, "150")
        else:
            self.length_label.config(text="Summary Length (sentences):")
            self.length_entry.delete(0, tk.END)
            self.length_entry.insert(0, "3")

    def _handle_file_upload(self):       #Handle file uploads from various formats
        file_types = [
            ("Text Files", "*.txt"),
            ("Word Documents", "*.docx"),
            ("PDF Files", "*.pdf")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if file_path:
            try:
                content = self._read_file(file_path)
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("File Error",
                                     f"Failed to read file:\n{str(e)}")

    def _read_file(self, path):        #Read content from different file formats
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

    def _generate_summary(self):        #Main summary generation
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

    def _extractive_summary(self, text):        #Generate extractive summary using NLTK
        try:
            num_sentences = int(self.length_entry.get())
            if num_sentences <= 0:
                raise ValueError("Number of sentences must be positive")

            expanded = self._expand_contractions(text)
            sentences = sent_tokenize(text)

            if len(sentences) <= num_sentences:
                return text

            # Calculate word frequencies
            words = [
                w.lower() for w in word_tokenize(expanded)
                if w.isalnum() and w not in stopwords.words('english')
            ]
            freq_dist = FreqDist(words)

            # Score and select sentences
            scores = []
            for idx, sent in enumerate(sentences):
                sent_words = [
                    w.lower() for w in word_tokenize(self._expand_contractions(sent))
                    if w.isalnum()
                ]
                score = sum(freq_dist[w] for w in sent_words) / \
                    len(sent_words) if sent_words else 0
                scores.append((idx, sent, score))

            # Select and order sentences
            sorted_sentences = sorted(scores, key=lambda x: x[2], reverse=True)
            selected = sorted(
                sorted_sentences[:num_sentences*2], key=lambda x: x[0])[:num_sentences]
            return ' '.join([s[1] for s in selected])

        except ValueError as e:
            raise ValueError(f"Extractive error: {str(e)}")

    def _abstractive_summary(self, text):        #Generate abstractive summary with precise length control
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
                truncation=True
            )[0]['summary_text']

            return self._process_abstractive_output(summary, target_words)
        except ValueError:
            raise ValueError("Invalid word limit")

    def _process_abstractive_output(self, summary, target):    #Post-process abstractive summary to meet word limit
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

    def _expand_contractions(self, text):       #Expand contractions in text
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
