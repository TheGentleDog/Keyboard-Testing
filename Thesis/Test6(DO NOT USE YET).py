import tkinter as tk
from tkinter import ttk
from collections import defaultdict
import json
import os

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

MAX_SUGGESTIONS = 5
LEARNING_FILE = "hmm_learning.json"

# Adaptive HMM storage
emission_counts = defaultdict(lambda: defaultdict(int))
transition_counts = defaultdict(lambda: defaultdict(int))

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

def replace_current_token(text, replacement):
    tokens = text.split()
    tokens[-1] = replacement
    return " ".join(tokens) + " "

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
# HYBRID PREDICTION
# ---------------------------------------------------------------------
def realtime_suggestions(token, prev_word):
    if not token:
        return []
    candidates = []
    for shortcut, words in SHORTCUT_DICT.items():
        sim = ngram_similarity(token, shortcut)
        if sim > 0:
            for word in words:
                dist = levenshtein(token, shortcut)
                hmm = adaptive_hmm_score(prev_word, token, word)
                final_score = (0.6 * sim) + (0.4 * hmm) - (0.1 * dist)
                candidates.append((word, final_score))
    candidates.sort(key=lambda x: -x[1])
    results, seen = [], set()
    for word, _ in candidates:
        if word not in seen:
            results.append(word)
            seen.add(word)
        if len(results) >= MAX_SUGGESTIONS:
            break
    return results

# ---------------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------------
class FilipinoIME(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filipino Shortcut Keyboard IME")
        self.geometry("1400x900")
        self.text = tk.StringVar()
        load_learning()
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Large.TButton", font=("Segoe UI", 16), padding=15)
        style.configure("Keyboard.TButton", font=("Segoe UI", 20), padding=10)
        
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(0, weight=1)   # Text area
        main_container.grid_rowconfigure(1, weight=0)   # Suggestions label
        main_container.grid_rowconfigure(2, weight=0)   # Suggestions buttons
        main_container.grid_rowconfigure(3, weight=2)   # Keyboard
        
        # TEXT DISPLAY
        text_label = ttk.Label(main_container, text="Typed Text", font=("Segoe UI", 18, "bold"))
        text_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        text_frame = ttk.Frame(main_container)
        text_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        self.text_display = tk.Text(text_frame, font=("Segoe UI", 20), height=4,
                                    wrap="word", relief="solid", borderwidth=2)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=scrollbar.set)
        self.text_display.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.text_display.bind("<KeyRelease>", self.on_text_change)
        
        # SUGGESTIONS LABEL (Row 1)
        suggestions_label = ttk.Label(main_container, text="Suggestions", font=("Segoe UI", 18, "bold"))
        suggestions_label.grid(row=1, column=0, sticky="w", pady=(10, 5))
        
        # SUGGESTIONS BUTTONS CONTAINER (Row 2 - BELOW the label)
        self.suggestions_container = ttk.Frame(main_container)
        self.suggestions_container.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        
        # KEYBOARD
        keyboard_label = ttk.Label(main_container, text="Virtual Keyboard", font=("Segoe UI", 18, "bold"))
        keyboard_label.grid(row=3, column=0, sticky="w", pady=(10, 5))
        
        keyboard_frame = ttk.Frame(main_container)
        keyboard_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        
        self.create_keyboard(keyboard_frame)

    def on_text_change(self, event=None):
        self.text.set(self.text_display.get("1.0", "end-1c"))
        self.update_suggestions()

    def update_suggestions(self):
        # Clear existing suggestion buttons
        for w in self.suggestions_container.winfo_children():
            w.destroy()
        
        text = self.text.get()
        token = get_current_token(text)
        words = text.split()
        prev_word = words[-2] if len(words) > 1 else "<START>"
        suggestions = realtime_suggestions(token, prev_word)
        
        for word in suggestions:
            btn = ttk.Button(
                self.suggestions_container,
                text=word,
                command=lambda w=word: self.apply_suggestion(w),
                style="Large.TButton"
            )
            btn.pack(side="left", padx=8, pady=5, ipadx=20, ipady=10)

    def apply_suggestion(self, word):
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        words = text.split()
        prev = words[-2] if len(words) > 1 else "<START>"
        emission_counts[word][token] += 1
        transition_counts[prev][word] += 1
        if token:
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            end_index = "end-1c"
            self.text_display.delete(start_index, end_index)
            self.text_display.insert(start_index, word + " ")
        self.update_suggestions()

    def insert_char(self, char):
        self.text_display.insert("insert", char)
        self.on_text_change()

    def backspace(self):
        self.text_display.delete("insert-1c")
        self.on_text_change()

    def create_keyboard(self, parent):
        kb_frame = ttk.Frame(parent)
        kb_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # TOP ROW - Numbers with Backspace next to 0
        row_num = ttk.Frame(kb_frame)
        row_num.pack(pady=5)
        for ch in "1234567890":
            btn = ttk.Button(row_num, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c), width=3)
            btn.pack(side="left", padx=2, pady=2)
        
        # Backspace next to 0
        backspace_btn = ttk.Button(row_num, text="⌫", 
                                  style="Keyboard.TButton",
                                  command=self.backspace)
        backspace_btn.pack(side="left", padx=2, pady=2, ipadx=15)
        
        # FIRST LETTER ROW
        row1 = ttk.Frame(kb_frame)
        row1.pack(pady=5)
        for ch in "qwertyuiop":
            btn = ttk.Button(row1, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c), width=3)
            btn.pack(side="left", padx=2, pady=2)
        
        # SECOND LETTER ROW with Enter next to L
        row2 = ttk.Frame(kb_frame)
        row2.pack(pady=5)
        ttk.Label(row2, width=2).pack(side="left")
        for ch in "asdfghjkl":
            btn = ttk.Button(row2, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c), width=3)
            btn.pack(side="left", padx=2, pady=2)
        
        # Enter next to L
        enter_btn = ttk.Button(row2, text="↵", 
                              style="Keyboard.TButton",
                              command=lambda: self.insert_char("\n"))
        enter_btn.pack(side="left", padx=2, pady=2, ipadx=15)
        
        # THIRD LETTER ROW
        row3 = ttk.Frame(kb_frame)
        row3.pack(pady=5)
        ttk.Label(row3, width=4).pack(side="left")
        for ch in "zxcvbnm":
            btn = ttk.Button(row3, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c), width=3)
            btn.pack(side="left", padx=2, pady=2)
        
        # BOTTOM ROW - JUST SPACE BAR
        bottom_frame = ttk.Frame(kb_frame)
        bottom_frame.pack(pady=15, fill="x", expand=True)
        
        # SPACE BAR - WIDE CENTERED
        space_btn = ttk.Button(bottom_frame, text="SPACE", 
                              style="Keyboard.TButton",
                              command=lambda: self.insert_char(" "))
        space_btn.pack(expand=True, fill="x", padx=100, pady=5, ipady=25)

    def on_close(self):
        save_learning()
        self.destroy()

# ---------------------------------------------------------------------
# RUN APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app = FilipinoIME()
    app.mainloop()