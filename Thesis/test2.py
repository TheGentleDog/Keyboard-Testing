import tkinter as tk

# -------------------------------
# Training data
# -------------------------------
TRAINING_DATA = [
    ("kng", "kung"),
    ("kc", "kasi"),
    ("mron", "meron"),
    ("mrn", "meron"),
    ("u", "ikaw"),
    ("sya", "siya"),
    ("d2", "dito"),
]
MAX_NGRAM = 2

# -------------------------------
# 1. Rule extraction
# -------------------------------
def extract_rules(training_data, max_ngram=2):
    rules = dict()
    for shortcut, correct in training_data:
        s_len, c_len = len(shortcut), len(correct)
        max_len = min(max_ngram, s_len, c_len)
        for n in range(1, max_len + 1):
            for i in range(s_len - n + 1):
                sub_s = shortcut[i:i + n]
                for j in range(c_len - n + 1):
                    sub_c = correct[j:j + n]
                    if sub_s not in rules:
                        rules[sub_s] = set()
                    rules[sub_s].add(sub_c)
    return rules

# -------------------------------
# 2. Candidate generation
# -------------------------------
def generate_candidates(word, rules):
    candidates = set([word])
    changed = True
    while changed:
        changed = False
        new_candidates = set()
        for cand in candidates:
            applied = False
            for sub in rules:
                if sub in cand:
                    for repl in rules[sub]:
                        new_word = cand.replace(sub, repl)
                        if new_word not in candidates:
                            new_candidates.add(new_word)
                            applied = True
            if not applied:
                new_candidates.add(cand)
            if applied:
                changed = True
        candidates = new_candidates
    return sorted(candidates)

# -------------------------------
# 3. Damerau-Levenshtein distance
# -------------------------------
def damerau_levenshtein(s1, s2):
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1
    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost,
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)
    return d[len1 - 1, len2 - 1]

def rank_candidates(word, candidates):
    return sorted(candidates, key=lambda x: damerau_levenshtein(word, x))

# -------------------------------
# 4. Normalization
# -------------------------------
def get_current_token(text):
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]

def normalize_token(token, rules):
    if not token:
        return []
    candidates = generate_candidates(token, rules)
    ranked = rank_candidates(token, candidates)
    return ranked[:5]  # top 5 suggestions

# -------------------------------
# 5. GUI
# -------------------------------
class FilipinoIME(tk.Tk):
    def __init__(self, rules):
        super().__init__()
        self.rules = rules
        self.title("Filipino Rule-Based IME")
        self.geometry("600x250")

        # Input box
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(self, textvariable=self.input_var, font=("Arial", 16))
        self.input_entry.pack(fill=tk.X, padx=10, pady=10)
        self.input_entry.focus_set()
        self.input_entry.bind("<KeyRelease>", self.on_key_release)

        # Suggestion label
        self.suggestion_label = tk.Label(self, text="", font=("Arial", 14), fg="blue")
        self.suggestion_label.pack(fill=tk.X, padx=10, pady=5)

        # Bind number keys to select suggestions
        for i in range(1, 6):
            self.bind(str(i), self.select_suggestion)

    def on_key_release(self, event=None):
        text = self.input_var.get()
        token = get_current_token(text)
        suggestions = normalize_token(token, self.rules)
        self.suggestion_label.config(text=" | ".join([f"{i+1}:{s}" for i, s in enumerate(suggestions)]))

    def select_suggestion(self, event):
        index = int(event.char) - 1
        text = self.input_var.get()
        token = get_current_token(text)
        suggestions = normalize_token(token, self.rules)
        if index < len(suggestions):
            # Replace token with selected suggestion
            new_text = text[:-(len(token))] + suggestions[index] + " "
            self.input_var.set(new_text)
            self.input_entry.icursor(tk.END)
            self.on_key_release()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    rules = extract_rules(TRAINING_DATA, MAX_NGRAM)
    app = FilipinoIME(rules)
    app.mainloop()
