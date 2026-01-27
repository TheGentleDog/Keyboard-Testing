import tkinter as tk
from tkinter import ttk

# -----------------------------
# Shortcut dictionary
# -----------------------------
SHORTCUT_DICT = {
    "kng": ["kung"],
    "kc": ["kasi"],
    "ksi": ["kasi"],
    "mron": ["meron"],
    "mrn": ["meron"],
    "d2": ["dito"],
    "dn": ["doon"],
    "jan": ["diyan"],
    "san": ["saan"],
    "tlg": ["talaga"],
    "pde": ["pwede"],
    "pd": ["pwede"],
    "aq": ["ako"],
    "xa": ["siya"],
    "sya": ["siya"],
    "u": ["ikaw", "oo"],
    "lng": ["lang"],
    "nmn": ["naman"],
    "sna": ["sana"],
    "s": ["sa"],
    "sry": ["sorry"],
    "syo": ["sayo"],
    "hbd": ["Happy Birthday"],

}

NGRAM_SIZE = 2
MAX_SUGGESTIONS = 5


# -----------------------------
# Text utilities
# -----------------------------
def get_current_token(text):
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]


def replace_current_token(text, replacement):
    tokens = text.split()
    tokens[-1] = replacement
    return " ".join(tokens) + " "


# -----------------------------
# Similarity functions
# -----------------------------
def char_ngrams(word, n=2):
    return {word[i:i+n] for i in range(len(word) - n + 1)} if len(word) >= n else {word}


def ngram_similarity(a, b, n=2):
    na = char_ngrams(a, n)
    nb = char_ngrams(b, n)
    if not na or not nb:
        return 0
    return len(na & nb) / len(na | nb)


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


def realtime_suggestions(token):
    if not token:
        return []

    candidates = []
    for shortcut, words in SHORTCUT_DICT.items():
        sim = ngram_similarity(token, shortcut)
        if sim > 0:
            for word in words:
                dist = levenshtein(token, shortcut)
                candidates.append((word, sim, dist))

    candidates.sort(key=lambda x: (-x[1], x[2]))

    results = []
    seen = set()
    for word, _, _ in candidates:
        if word not in seen:
            results.append(word)
            seen.add(word)
        if len(results) >= MAX_SUGGESTIONS:
            break
    return results


# -----------------------------
# GUI Application
# -----------------------------
class FilipinoIME(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filipino Shortcut Keyboard IME")
        self.geometry("700x500")

        self.text = tk.StringVar()

        self.create_widgets()
        self.update_suggestions()

    def create_widgets(self):
        ttk.Label(self, text="Typed Text").pack(anchor="w", padx=10)

        self.entry = ttk.Entry(self, textvariable=self.text, font=("Segoe UI", 14))
        self.entry.pack(fill="x", padx=10, pady=5)
        self.entry.bind("<KeyRelease>", lambda e: self.update_suggestions())

        self.suggestion_frame = ttk.Frame(self)
        self.suggestion_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self, text="Suggestions").pack(anchor="w", padx=10)
        self.suggestions_container = ttk.Frame(self)
        self.suggestions_container.pack(fill="x", padx=10)

        ttk.Label(self, text="Virtual Keyboard").pack(anchor="w", padx=10, pady=5)
        self.create_keyboard()

    def update_suggestions(self):
        for widget in self.suggestions_container.winfo_children():
            widget.destroy()

        token = get_current_token(self.text.get())
        suggestions = realtime_suggestions(token)

        for word in suggestions:
            btn = ttk.Button(
                self.suggestions_container,
                text=word,
                command=lambda w=word: self.apply_suggestion(w)
            )
            btn.pack(side="left", padx=3)

    def apply_suggestion(self, word):
        new_text = replace_current_token(self.text.get(), word)
        self.text.set(new_text)
        self.entry.icursor(tk.END)
        self.update_suggestions()

    def insert_char(self, char):
        self.text.set(self.text.get() + char)
        self.update_suggestions()

    def backspace(self):
        self.text.set(self.text.get()[:-1])
        self.update_suggestions()

    def create_keyboard(self):
        keys = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ]

        kb = ttk.Frame(self)
        kb.pack()

        for row in keys:
            r = ttk.Frame(kb)
            r.pack()
            for ch in row:
                ttk.Button(r, text=ch, width=4,
                           command=lambda c=ch: self.insert_char(c)).pack(side="left", padx=2, pady=2)

        bottom = ttk.Frame(kb)
        bottom.pack(pady=5)

        ttk.Button(bottom, text="Space", width=15,
                   command=lambda: self.insert_char(" ")).pack(side="left", padx=5)

        ttk.Button(bottom, text="Backspace", width=12,
                   command=self.backspace).pack(side="left", padx=5)

        ttk.Button(bottom, text="Enter", width=10,
                   command=lambda: self.insert_char("\n")).pack(side="left", padx=5)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app = FilipinoIME()
    app.mainloop()
