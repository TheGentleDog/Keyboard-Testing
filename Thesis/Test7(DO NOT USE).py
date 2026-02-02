import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
import re
from datasets import load_dataset

# Shortcut dictionary
SHORTCUT_DICT = {
    "kng": ["kung"], "kc": ["kasi"], "ksi": ["kasi"], "mron": ["meron"],
    "mrn": ["meron"], "d2": ["dito"], "dn": ["doon"], "jan": ["diyan"],
    "san": ["saan"], "tlg": ["talaga"], "pde": ["pwede"], "pd": ["pwede"],
    "aq": ["ako"], "xa": ["siya"], "sya": ["siya"], "u": ["ikaw", "oo"],
    "lng": ["lang"], "nmn": ["naman"], "sna": ["sana"], "s": ["sa"],
    "sry": ["sorry"], "syo": ["sayo"], "hbd": ["happy birthday"],
    "nde": ["hinde", "hindi"],
}

MAX_SUGGESTIONS = 8
LEARNING_FILE = "hmm_learning.json"
DATASET_CACHE_FILE = "filipino_vocabulary.json"

# Adaptive HMM storage
emission_counts = defaultdict(lambda: defaultdict(int))
transition_counts = defaultdict(lambda: defaultdict(int))

# ---------------------------------------------------------------------
# DATASET VOCABULARY LOADER (SIMPLIFIED)
# ---------------------------------------------------------------------
class FilipinoVocabulary:
    def __init__(self):
        self.vocabulary = set()
        self.word_frequencies = Counter()
        self.context_pairs = defaultdict(lambda: defaultdict(int))
        self.common_words = []
        
        print("Loading Filipino vocabulary...")
        self.load_fallback_vocabulary()
        print(f"✓ Loaded {len(self.vocabulary)} Filipino words")
        print(f"✓ Top 10 common words: {self.common_words[:10]}")
    
    def load_fallback_vocabulary(self):
        """Load a basic Filipino vocabulary"""
        fallback_words = [
            "ang", "ng", "sa", "mga", "na", "ay", "at", "si", "ni", "kay",
            "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
            "ito", "iyan", "iyon", "dito", "doon", "diyan",
            "may", "wala", "hindi", "oo", "bakit", "paano", "kailan", "saan",
            "kung", "dahil", "pero", "o", "at", "para", "tungkol",
            "malaki", "maliit", "maganda", "pangit", "masaya", "malungkot",
            "kumain", "uminom", "matulog", "maglaro", "magbasa", "magsulat",
            "bahay", "eskwela", "trabaho", "pamilya", "kaibigan", "pagkain",
            "tubig", "hangin", "araw", "buwan", "bituin", "langit", "lupa"
        ]
        
        self.vocabulary = set(fallback_words)
        self.word_frequencies = Counter({word: 10 for word in fallback_words})
        self.common_words = fallback_words
        
        # Add context pairs (simplified)
        for i in range(len(fallback_words) - 1):
            self.context_pairs[fallback_words[i]][fallback_words[i+1]] += 1
    
    def get_predictive_suggestions(self, previous_word="", max_results=6):
        """Get predictive suggestions based on previous word"""
        suggestions = []
        
        # 1. Context-based predictions
        if previous_word and previous_word in self.context_pairs:
            context_words = self.context_pairs[previous_word]
            for next_word, count in sorted(context_words.items(), 
                                          key=lambda x: x[1], reverse=True)[:max_results]:
                suggestions.append(next_word)
        
        # 2. Most common Filipino words as fallback
        if not suggestions or len(suggestions) < max_results:
            for word in self.common_words[:max_results]:
                if not previous_word or word != previous_word:
                    if word not in suggestions:
                        suggestions.append(word)
        
        return suggestions[:max_results]

# Initialize vocabulary
vocab = FilipinoVocabulary()

# ---------------------------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------------------------
def load_learning():
    if not os.path.exists(LEARNING_FILE):
        return
    with open(LEARNING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for w, obs in data.get("emissions", {}).items():
            for o, c in obs.items():
                emission_counts[w][o] = c
        for p, nxt in data.get("transitions", {}).items():
            for n, c in nxt.items():
                transition_counts[p][n] = c

def save_learning():
    data = {"emissions": emission_counts, "transitions": transition_counts}
    with open(LEARNING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------
def get_current_token(text):
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]

def get_previous_word(text):
    """Get the word before current token for context"""
    words = text.strip().split()
    if len(words) < 2:
        return ""
    return words[-2].lower()

# ---------------------------------------------------------------------
# SIMILARITY FUNCTIONS
# ---------------------------------------------------------------------
def char_ngrams(word, n=2):
    return {word[i:i+n] for i in range(len(word)-n+1)} if len(word) >= n else {word}

def ngram_similarity(a, b):
    na = char_ngrams(a)
    nb = char_ngrams(b)
    return len(na & nb) / len(na | nb) if na and nb else 0

def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = range(len(b) + 1)
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            delete = curr[j] + 1
            rep = prev[j] + (ca != cb)
            curr.append(min(ins, delete, rep))
        prev = curr
    return prev[-1]

# ---------------------------------------------------------------------
# ADAPTIVE HMM SCORING
# ---------------------------------------------------------------------
def adaptive_hmm_score(prev_word, token, candidate):
    e_total = sum(emission_counts[candidate].values()) + 1
    t_total = sum(transition_counts[prev_word].values()) + 1
    emission = (emission_counts[candidate][token] + 1) / e_total
    transition = (transition_counts[prev_word][candidate] + 1) / t_total
    return emission * transition

# ---------------------------------------------------------------------
# HYBRID PREDICTION WITH DATASET VOCABULARY
# ---------------------------------------------------------------------
def realtime_suggestions(token, prev_word):
    if not token:
        return []
    
    candidates = []
    
    # 1. SHORTCUT EXPANSION
    for shortcut, words in SHORTCUT_DICT.items():
        if token:
            sim = ngram_similarity(token, shortcut)
            if sim > 0.3:
                for word in words:
                    dist = levenshtein(token, shortcut)
                    hmm = adaptive_hmm_score(prev_word, token, word)
                    final_score = (0.6 * sim) + (0.4 * hmm) - (0.1 * dist)
                    candidates.append((word, final_score, "shortcut"))
    
    # 2. DATASET VOCABULARY SUGGESTIONS (word completion)
    for word in vocab.vocabulary:
        if word.startswith(token.lower()):
            freq = vocab.word_frequencies.get(word, 1)
            score = 0.5 + min(freq / 20, 0.5)
            candidates.append((word, score, "vocabulary"))
    
    # Sort by score and remove duplicates
    candidates.sort(key=lambda x: -x[1])
    results, seen = [], set()
    
    for word, score, category in candidates:
        if word not in seen:
            # Capitalize if the token was capitalized
            display_word = word.capitalize() if token and token[0].isupper() else word
            results.append(display_word)
            seen.add(word)
        if len(results) >= MAX_SUGGESTIONS:
            break
    
    return results

# ---------------------------------------------------------------------
# GUI APPLICATION - FIXED VERSION
# ---------------------------------------------------------------------
class FilipinoIME(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("🇵🇭 Filipino Keyboard with Predictive Buttons")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Configure main grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Initialize
        load_learning()
        self.create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Tab>", lambda e: self.auto_complete())
        
        # Update suggestions initially
        self.update_suggestions()

    def create_widgets(self):
        # Main container using Frame
        main_container = ttk.Frame(self, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure main container grid
        main_container.grid_rowconfigure(0, weight=0)   # Title
        main_container.grid_rowconfigure(1, weight=1)   # Text area
        main_container.grid_rowconfigure(2, weight=0)   # Completion label
        main_container.grid_rowconfigure(3, weight=0)   # Completion buttons
        main_container.grid_rowconfigure(4, weight=0)   # Predictive label
        main_container.grid_rowconfigure(5, weight=0)   # Predictive buttons
        main_container.grid_rowconfigure(6, weight=0)   # Keyboard label
        main_container.grid_rowconfigure(7, weight=0)   # Keyboard
        main_container.grid_columnconfigure(0, weight=1)
        
        # TITLE
        title_label = ttk.Label(
            main_container, 
            text="🇵🇭 Filipino Keyboard with Predictive Buttons",
            font=("Segoe UI", 18, "bold")
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # TEXT AREA
        text_frame = ttk.LabelFrame(main_container, text="Text Input", padding="10")
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        
        self.text_display = tk.Text(
            text_frame, 
            font=("Segoe UI", 16),
            wrap="word",
            relief="solid",
            borderwidth=1,
            height=8
        )
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=scrollbar.set)
        
        self.text_display.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.text_display.bind("<KeyRelease>", self.on_text_change)
        self.text_display.bind("<space>", lambda e: self.force_update_suggestions())
        
        # WORD COMPLETION SUGGESTIONS LABEL
        suggestions_label = ttk.Label(
            main_container,
            text="Word Completion Suggestions:",
            font=("Segoe UI", 14, "bold")
        )
        suggestions_label.grid(row=2, column=0, sticky="w", pady=(5, 5))
        
        # WORD COMPLETION SUGGESTIONS CONTAINER
        self.completion_frame = ttk.Frame(main_container)
        self.completion_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.completion_frame.grid_columnconfigure(0, weight=1)
        
        # Create completion buttons container
        self.completion_container = ttk.Frame(self.completion_frame)
        self.completion_container.grid(row=0, column=0, sticky="w")
        
        # PREDICTIVE SUGGESTIONS LABEL
        predictive_label = ttk.Label(
            main_container,
            text="Predictive Suggestions (Next Word):",
            font=("Segoe UI", 14, "bold")
        )
        predictive_label.grid(row=4, column=0, sticky="w", pady=(5, 5))
        
        # PREDICTIVE SUGGESTIONS CONTAINER
        self.predictive_frame = ttk.Frame(main_container)
        self.predictive_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.predictive_frame.grid_columnconfigure(0, weight=1)
        
        # Create predictive buttons container
        self.predictive_container = ttk.Frame(self.predictive_frame)
        self.predictive_container.grid(row=0, column=0, sticky="w")
        
        # KEYBOARD LABEL
        keyboard_label = ttk.Label(
            main_container,
            text="Virtual Keyboard:",
            font=("Segoe UI", 14, "bold")
        )
        keyboard_label.grid(row=6, column=0, sticky="w", pady=(5, 5))
        
        # KEYBOARD FRAME
        keyboard_frame = ttk.Frame(main_container)
        keyboard_frame.grid(row=7, column=0, sticky="nsew", pady=(0, 5))
        
        # Create keyboard inside this frame
        self.create_keyboard(keyboard_frame)
        
        # Add a status bar
        self.status_bar = ttk.Label(main_container, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=8, column=0, sticky="ew", pady=(5, 0))

    def on_text_change(self, event=None):
        current_text = self.text_display.get("1.0", "end-1c")
        self.update_suggestions()
        self.status_bar.config(text=f"Text length: {len(current_text)} characters")

    def force_update_suggestions(self):
        """Force update suggestions after space is pressed"""
        self.after(10, self.update_suggestions)

    def update_suggestions(self):
        """Update both completion and predictive suggestions"""
        # Clear existing suggestion buttons
        for widget in self.completion_container.winfo_children():
            widget.destroy()
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        prev_word = get_previous_word(text)
        
        # 1. WORD COMPLETION SUGGESTIONS (for current token)
        if token:
            completion_suggestions = realtime_suggestions(token, prev_word)
            if completion_suggestions:
                for word in completion_suggestions[:6]:
                    btn = ttk.Button(
                        self.completion_container,
                        text=word,
                        command=lambda w=word: self.apply_suggestion(w),
                        style="Suggestion.TButton"
                    )
                    btn.pack(side="left", padx=3, pady=3, ipadx=8, ipady=3)
            else:
                no_suggestions = ttk.Label(
                    self.completion_container,
                    text="No word completion suggestions",
                    font=("Segoe UI", 10),
                    foreground="gray"
                )
                no_suggestions.pack()
        else:
            # Show a placeholder when no token
            placeholder = ttk.Label(
                self.completion_container,
                text="Start typing to see word completion suggestions",
                font=("Segoe UI", 10, "italic"),
                foreground="gray"
            )
            placeholder.pack()
        
        # 2. PREDICTIVE SUGGESTIONS (next word predictions)
        # Show predictive suggestions only when not typing a word (at end of word)
        if not token or text.endswith(" "):
            predictive_words = vocab.get_predictive_suggestions(prev_word, max_results=6)
            
            if predictive_words:
                for word in predictive_words[:6]:
                    # Capitalize if the previous word was capitalized
                    display_word = word.capitalize() if prev_word and prev_word[0].isupper() else word
                    
                    btn = ttk.Button(
                        self.predictive_container,
                        text=display_word,
                        command=lambda w=display_word: self.apply_predictive_suggestion(w),
                        style="Predictive.TButton"
                    )
                    btn.pack(side="left", padx=5, pady=5, ipadx=12, ipady=5)
            else:
                # Show common Filipino words as fallback
                for word in vocab.common_words[:6]:
                    btn = ttk.Button(
                        self.predictive_container,
                        text=word,
                        command=lambda w=word: self.apply_predictive_suggestion(w),
                        style="Predictive.TButton"
                    )
                    btn.pack(side="left", padx=5, pady=5, ipadx=12, ipady=5)
        else:
            # When typing a word, show a message
            placeholder = ttk.Label(
                self.predictive_container,
                text="Finish typing current word for predictive suggestions",
                font=("Segoe UI", 10, "italic"),
                foreground="gray"
            )
            placeholder.pack()

    def apply_suggestion(self, word):
        """Apply word completion suggestion"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        prev_word = get_previous_word(text)
        
        if token:
            # Update learning models
            emission_counts[word.lower()][token.lower()] += 1
            if prev_word:
                transition_counts[prev_word.lower()][word.lower()] += 1
            
            # Apply the suggestion
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            
            self.text_display.delete(start_index, "end-1c")
            self.text_display.insert(start_index, word + " ")
        else:
            self.text_display.insert("end", word + " ")
        
        # Force update suggestions immediately
        self.update_suggestions()
        
        self.status_bar.config(text=f"Applied suggestion: '{word}'")
    
    def apply_predictive_suggestion(self, word):
        """Apply predictive suggestion (next word)"""
        text = self.text_display.get("1.0", "end-1c")
        prev_word = get_previous_word(text)
        
        # Update learning models
        if prev_word:
            transition_counts[prev_word.lower()][word.lower()] += 1
        
        # Always add a space before the predictive word if needed
        if text and not text.endswith(" "):
            self.text_display.insert("end", " ")
        
        self.text_display.insert("end", word + " ")
        
        # Force update suggestions immediately
        self.update_suggestions()
        
        self.status_bar.config(text=f"Applied predictive word: '{word}'")

    def auto_complete(self):
        """Auto-complete the current word"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        prev_word = get_previous_word(text)
        
        if token and len(token) >= 2:
            suggestions = realtime_suggestions(token, prev_word)
            if suggestions:
                self.apply_suggestion(suggestions[0])
                self.status_bar.config(text=f"Auto-completed to '{suggestions[0]}'")

    def insert_char(self, char):
        self.text_display.insert("insert", char)
        self.text_display.event_generate("<KeyRelease>")

    def backspace(self):
        self.text_display.delete("insert-1c")
        self.text_display.event_generate("<KeyRelease>")
    
    def space(self):
        self.text_display.insert("insert", " ")
        self.text_display.event_generate("<KeyRelease>")
    
    def enter(self):
        self.text_display.insert("insert", "\n")
        self.text_display.event_generate("<KeyRelease>")

    def create_keyboard(self, parent):
        # Configure styles
        style = ttk.Style()
        style.configure("Keyboard.TButton", font=("Segoe UI", 12), padding=6)
        style.configure("Predictive.TButton", font=("Segoe UI", 11), padding=8, background="#e6f3ff")
        style.configure("Suggestion.TButton", font=("Segoe UI", 10), padding=4, background="#f0f0f0")
        
        # Use grid for better control
        parent.grid_columnconfigure(0, weight=1)
        
        # Function keys row
        func_row = ttk.Frame(parent)
        func_row.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Auto-complete button
        auto_btn = ttk.Button(func_row, text="TAB: Auto-Complete", 
                            style="Keyboard.TButton",
                            command=self.auto_complete)
        auto_btn.pack(side="left", padx=2, ipadx=5, expand=True, fill="x")
        
        # Clear button
        clear_btn = ttk.Button(func_row, text="Clear Text", 
                             style="Keyboard.TButton",
                             command=lambda: [self.text_display.delete("1.0", "end"), 
                                            self.update_suggestions()])
        clear_btn.pack(side="left", padx=2, ipadx=5, expand=True, fill="x")
        
        # TOP ROW - Numbers with Backspace
        row_num = ttk.Frame(parent)
        row_num.grid(row=1, column=0, sticky="ew", pady=2)
        
        for ch in "1234567890":
            btn = ttk.Button(row_num, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # Backspace
        backspace_btn = ttk.Button(row_num, text="⌫", 
                                  style="Keyboard.TButton",
                                  command=self.backspace)
        backspace_btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # FIRST LETTER ROW
        row1 = ttk.Frame(parent)
        row1.grid(row=2, column=0, sticky="ew", pady=2)
        
        for ch in "qwertyuiop":
            btn = ttk.Button(row1, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # SECOND LETTER ROW
        row2 = ttk.Frame(parent)
        row2.grid(row=3, column=0, sticky="ew", pady=2)
        
        # Add left padding
        left_pad = ttk.Frame(row2, width=20)
        left_pad.pack(side="left")
        
        for ch in "asdfghjkl":
            btn = ttk.Button(row2, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # Enter button
        enter_btn = ttk.Button(row2, text="↵", 
                              style="Keyboard.TButton",
                              command=self.enter)
        enter_btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # THIRD LETTER ROW
        row3 = ttk.Frame(parent)
        row3.grid(row=4, column=0, sticky="ew", pady=2)
        
        # Add more left padding
        left_pad = ttk.Frame(row3, width=40)
        left_pad.pack(side="left")
        
        for ch in "zxcvbnm":
            btn = ttk.Button(row3, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # SPACE BAR ROW - IMPORTANT: This is the space button
        space_row = ttk.Frame(parent)
        space_row.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        
        space_btn = ttk.Button(space_row, text="SPACE", 
                              style="Keyboard.TButton",
                              command=self.space)
        space_btn.pack(ipadx=50, ipady=10, expand=True, fill="x")

    def on_close(self):
        save_learning()
        self.destroy()

# ---------------------------------------------------------------------
# RUN APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Filipino Keyboard with Predictive Buttons...")
    app = FilipinoIME()
    app.mainloop()