import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
import pickle

# Configuration
MAX_SUGGESTIONS = 5
NGRAM_CACHE_FILE = "ngram_model_standalone.pkl"
USER_LEARNING_FILE = "user_learning.json"
DATASET_FILE = "filipino_dataset.json"
MODEL_VERSION = "2.1_json_dataset"

# =============================================================================
# LOAD DATASET FROM JSON
# =============================================================================
def load_dataset():
    """Load Filipino vocabulary and corpus from JSON file"""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded dataset: {data['metadata']['description']}")
        print(f"  Version: {data['metadata']['version']}")
        print(f"  Words: {data['metadata']['total_words']}")
        print(f"  Phrases: {data['metadata']['total_phrases']}")
        
        words = []
        for category, word_list in data['vocabulary'].items():
            words.extend(word_list)
        
        shortcuts = data['shortcuts']
        corpus = data['communication_corpus']
        
        return words, shortcuts, corpus
        
    except FileNotFoundError:
        print(f"⚠ Warning: {DATASET_FILE} not found!")
        print("⚠ Using minimal fallback vocabulary...")
        return _get_fallback_data()
    except Exception as e:
        print(f"⚠ Error loading dataset: {e}")
        print("⚠ Using minimal fallback vocabulary...")
        return _get_fallback_data()

def _get_fallback_data():
    """Minimal fallback if JSON file is missing"""
    words = [
        "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
        "kumain", "uminom", "matulog", "maganda", "masaya",
        "oo", "hindi", "salamat", "sorry", "nalang", "lang",
        "naman", "kasi", "kung", "dito", "doon", "pwede",
        "sana", "talaga", "para", "wala", "may", "yung"
    ]
    shortcuts = {
        "lng": "lang", "nlng": "nalang", "nmn": "naman", 
        "ksi": "kasi", "kng": "kung", "khit": "kahit",
        "d2": "dito", "dn": "doon", "pde": "pwede", "pwd": "pwede",
        "sna": "sana", "tlg": "talaga", "tlga": "talaga",
        "sya": "siya", "xa": "siya", "aq": "ako", "q": "ako",
        "nde": "hindi", "hnd": "hindi", "pra": "para",
        "wla": "wala", "my": "may", "ung": "yung", "un": "yun",
        "dba": "diba", "db": "diba", "bat": "bakit", "bkt": "bakit"
    }
    corpus = ["kumusta ka", "mabuti naman", "salamat", "ok lang"]
    return words, shortcuts, corpus

FILIPINO_WORDS, FILIPINO_SHORTCUTS, COMMUNICATION_CORPUS = load_dataset()

# =============================================================================
# DAMERAU-LEVENSHTEIN DISTANCE (Typo Tolerance)
# =============================================================================
def damerau_levenshtein_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    d = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1
    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i-1, j)] + 1,
                d[(i, j-1)] + 1,
                d[(i-1, j-1)] + cost,
            )
            if i > 0 and j > 0 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i-2, j-2)] + cost)
    return d[(len1-1, len2-1)]

# =============================================================================
# N-GRAM LANGUAGE MODEL
# =============================================================================
class NgramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.vocabulary = set()
        self.total_words = 0
        self.char_bigrams = defaultdict(Counter)
        self.char_trigrams = defaultdict(Counter)
        self.csv_shortcuts = {}
        self.user_shortcuts = {}
        self.user_shortcut_usage = Counter()
        self.new_words = set()
        self.word_usage_history = []
    
    def train_from_builtin(self):
        print("📚 Loading built-in Filipino vocabulary...")
        all_tokens = []
        for word in FILIPINO_WORDS:
            all_tokens.extend([word.lower()] * 10)
        print("📝 Processing communication corpus...")
        for phrase in COMMUNICATION_CORPUS:
            words = phrase.lower().split()
            for _ in range(50):
                all_tokens.extend(words)
        for shortcut, full_word in FILIPINO_SHORTCUTS.items():
            self.csv_shortcuts[shortcut] = full_word
            all_tokens.extend([full_word.lower()] * 5)
        essential_shortcuts = {
            "nlng": "nalang", "lng": "lang", "nmn": "naman",
            "ksi": "kasi", "kse": "kasi", "kng": "kung", "khit": "kahit",
            "d2": "dito", "dn": "doon", "dun": "doon",
            "pde": "pwede", "pwd": "pwede",
            "sna": "sana", "tlg": "talaga", "tlga": "talaga",
            "sya": "siya", "xa": "siya", "aq": "ako", "q": "ako",
            "nde": "hindi", "hnd": "hindi", "di": "hindi",
            "pra": "para", "wla": "wala", "my": "may",
            "ung": "yung", "un": "yun", "dba": "diba", "db": "diba",
            "bat": "bakit", "bkt": "bakit", "pno": "paano", "pano": "paano",
            "tyo": "tayo", "kyo": "kayo", "kmi": "kami", "cla": "sila",
            "ikw": "ikaw", "ako": "ako"
        }
        print(f"📌 Adding {len(essential_shortcuts)} essential shortcuts...")
        for shortcut, full_word in essential_shortcuts.items():
            self.csv_shortcuts[shortcut] = full_word
            all_tokens.extend([full_word.lower()] * 5)
        print(f"✓ Built-in words: {len(FILIPINO_WORDS)}")
        print(f"✓ Total shortcuts: {len(self.csv_shortcuts)}")
        print(f"✓ Total tokens: {len(all_tokens)}")
        self.vocabulary.update(all_tokens)
        print("\n🔨 Building n-grams...")
        for i, token in enumerate(all_tokens):
            self.unigrams[token] += 1
            self.total_words += 1
            self._build_char_ngrams(token)
            if i > 0:
                prev = all_tokens[i-1]
                self.bigrams[prev][token] += 1
            if i > 1:
                prev2 = all_tokens[i-2]
                prev1 = all_tokens[i-1]
                self.trigrams[(prev2, prev1)][token] += 1
        print(f"✓ Vocabulary: {len(self.vocabulary)} unique words")
    
    def _has_vowels(self, word):
        return any(c in 'aeiouAEIOU' for c in word)
    
    def _build_char_ngrams(self, word):
        if len(word) < 2:
            return
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            if i < len(word) - 2:
                next_char = word[i+2]
                self.char_bigrams[bigram][next_char] += 1
            else:
                self.char_bigrams[bigram]["<END>"] += 1
        for i in range(len(word) - 2):
            trigram = (word[i], word[i+1])
            if i < len(word) - 3:
                next_char = word[i+3]
                self.char_trigrams[trigram][next_char] += 1
    
    def get_char_level_completions(self, prefix, max_results=5):
        if len(prefix) < 1:
            return []
        min_length = 2 if len(prefix) == 1 else max(len(prefix) + 1, 3)
        candidates = [
            word for word in self.vocabulary 
            if word.startswith(prefix) 
            and len(word) >= min_length
            and word.isalpha()
            and self._has_vowels(word)
        ]
        scored = []
        for word in candidates:
            score = 0
            for i in range(len(prefix), len(word) - 1):
                if i >= 2:
                    bigram = word[i-2:i]
                    if bigram in self.char_bigrams:
                        next_char = word[i]
                        count = self.char_bigrams[bigram].get(next_char, 0)
                        total = sum(self.char_bigrams[bigram].values())
                        if total > 0:
                            score += count / total
            freq_score = self.unigrams.get(word, 0) / max(self.total_words, 1)
            final_score = score + (freq_score * 10)
            scored.append((word, final_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in scored[:max_results]]
    
    def resolve_shortcut(self, word):
        word_lower = word.lower()
        if word_lower in self.user_shortcuts:
            return self.user_shortcuts[word_lower]
        if word_lower in self.csv_shortcuts:
            return self.csv_shortcuts[word_lower]
        return word
    
    def get_all_shortcut_expansions(self, shortcut):
        shortcut_lower = shortcut.lower()
        expansions = []
        if shortcut_lower in self.user_shortcuts:
            expansions.append((self.user_shortcuts[shortcut_lower], 'user'))
        if shortcut_lower in self.csv_shortcuts:
            word = self.csv_shortcuts[shortcut_lower]
            if word not in [w for w, _ in expansions]:
                expansions.append((word, 'csv'))
        return expansions
    
    def learn_from_user_typing(self, typed_shortcut, selected_word):
        typed_shortcut = typed_shortcut.lower()
        selected_word = selected_word.lower()
        if len(typed_shortcut) < len(selected_word) - 1:
            if typed_shortcut not in self.user_shortcuts:
                self.user_shortcuts[typed_shortcut] = selected_word
                print(f"🎓 Learned: '{typed_shortcut}' → '{selected_word}'")
            self.user_shortcut_usage[typed_shortcut] += 1
            self.save_user_learning()
    
    def add_new_word(self, word):
        word_lower = word.lower()
        if word_lower not in self.vocabulary:
            self.vocabulary.add(word_lower)
            self.new_words.add(word_lower)
            self.unigrams[word_lower] = 1
            print(f"📝 New word: '{word}'")
            self.save_user_learning()
    
    def track_word_usage(self, word, context=None):
        word_lower = word.lower()
        self.unigrams[word_lower] += 1
        self.total_words += 1
        if context and len(context) >= 1:
            prev = context[-1].lower()
            prev = self.resolve_shortcut(prev)
            self.bigrams[prev][word_lower] += 1
        if context and len(context) >= 2:
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            self.trigrams[(prev2, prev1)][word_lower] += 1
        self.word_usage_history.append((word_lower, context))
        if len(self.word_usage_history) > 1000:
            self.word_usage_history.pop(0)
    
    def save_user_learning(self):
        user_data = {
            'user_shortcuts': self.user_shortcuts,
            'user_shortcut_usage': dict(self.user_shortcut_usage),
            'new_words': list(self.new_words),
            'word_usage_history': self.word_usage_history[-100:]
        }
        try:
            with open(USER_LEARNING_FILE, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Error saving: {e}")
    
    def load_user_learning(self):
        if not os.path.exists(USER_LEARNING_FILE):
            return
        try:
            with open(USER_LEARNING_FILE, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            self.user_shortcuts = user_data.get('user_shortcuts', {})
            self.user_shortcut_usage = Counter(user_data.get('user_shortcut_usage', {}))
            self.new_words = set(user_data.get('new_words', []))
            self.word_usage_history = user_data.get('word_usage_history', [])
            self.vocabulary.update(self.new_words)
            print(f"✓ User shortcuts: {len(self.user_shortcuts)}")
            print(f"✓ New words: {len(self.new_words)}")
        except Exception as e:
            print(f"⚠ Error loading: {e}")
    
    def save_cache(self):
        data = {
            'version': MODEL_VERSION,
            'unigrams': dict(self.unigrams),
            'bigrams': {k: dict(v) for k, v in self.bigrams.items()},
            'trigrams': {k: dict(v) for k, v in self.trigrams.items()},
            'char_bigrams': {k: dict(v) for k, v in self.char_bigrams.items()},
            'char_trigrams': {k: dict(v) for k, v in self.char_trigrams.items()},
            'vocabulary': list(self.vocabulary),
            'total_words': self.total_words,
            'csv_shortcuts': self.csv_shortcuts,
        }
        with open(NGRAM_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved to {NGRAM_CACHE_FILE}")
    
    def load_cache(self):
        if not os.path.exists(NGRAM_CACHE_FILE):
            return False
        try:
            with open(NGRAM_CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            cached_version = data.get('version', '1.0')
            if cached_version != MODEL_VERSION:
                print(f"⚠ Cache version mismatch, rebuilding...")
                return False
            self.unigrams = Counter(data['unigrams'])
            self.bigrams = defaultdict(Counter)
            for k, v in data['bigrams'].items():
                self.bigrams[k] = Counter(v)
            self.trigrams = defaultdict(Counter)
            for k, v in data['trigrams'].items():
                self.trigrams[k] = Counter(v)
            self.char_bigrams = defaultdict(Counter)
            if 'char_bigrams' in data:
                for k, v in data['char_bigrams'].items():
                    self.char_bigrams[k] = Counter(v)
            self.char_trigrams = defaultdict(Counter)
            if 'char_trigrams' in data:
                for k, v in data['char_trigrams'].items():
                    self.char_trigrams[k] = Counter(v)
            self.vocabulary = set(data['vocabulary'])
            self.total_words = data['total_words']
            self.csv_shortcuts = data.get('csv_shortcuts', {})
            print(f"✓ Loaded cache — Vocabulary: {len(self.vocabulary)}, Shortcuts: {len(self.csv_shortcuts)}")
            return True
        except Exception as e:
            print(f"⚠ Error: {e}")
            return False
    
    def get_word_probability(self, word, context=None):
        word = word.lower()
        alpha = 0.1
        if context is None or len(context) == 0:
            count = self.unigrams.get(word, 0)
            vocab_size = len(self.vocabulary)
            return (count + alpha) / (self.total_words + alpha * vocab_size)
        elif len(context) == 1:
            prev = context[0].lower()
            prev = self.resolve_shortcut(prev)
            count = self.bigrams[prev].get(word, 0)
            prev_count = self.unigrams.get(prev, 0)
            vocab_size = len(self.vocabulary)
            if prev_count == 0:
                return self.get_word_probability(word)
            return (count + alpha) / (prev_count + alpha * vocab_size)
        else:
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            trigram_context = (prev2, prev1)
            count = self.trigrams[trigram_context].get(word, 0)
            context_count = sum(self.trigrams[trigram_context].values())
            vocab_size = len(self.vocabulary)
            if context_count == 0:
                return self.get_word_probability(word, [prev1])
            return (count + alpha) / (context_count + alpha * vocab_size)
    
    def get_completion_suggestions(self, prefix, context=None, max_results=8):
        prefix = prefix.lower()
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        if all_expansions:
            for full_word, source in all_expansions:
                priority_multiplier = 10.0 if source == 'user' else 8.0
                shortcut_candidates.append((full_word, False, True, priority_multiplier))
        candidates = []
        min_word_length = 2
        exact_matches = [
            word for word in self.vocabulary 
            if word.startswith(prefix) 
            and len(word) >= min_word_length
            and word.isalpha()
            and self._has_vowels(word)
        ]
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        for word in exact_matches:
            if word in char_completions:
                candidates.append((word, True, False, 2.0))
            else:
                candidates.append((word, True, False, 1.0))
        fuzzy_candidates = []
        if len(prefix) >= 2:
            for word in self.vocabulary:
                if word in exact_matches:
                    continue
                if len(word) < min_word_length:
                    continue
                if not word.isalpha():
                    continue
                if not self._has_vowels(word):
                    continue
                if prefix in word:
                    position = word.index(prefix)
                    fuzzy_candidates.append((word, False, False, 0.5 / (position + 1)))
                    continue
                if abs(len(word) - len(prefix)) <= 2:
                    distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                    if distance == 1:
                        similarity = 1.0 / (1.0 + distance)
                        fuzzy_candidates.append((word, False, False, similarity * 0.2))
        all_candidates = shortcut_candidates + candidates + fuzzy_candidates
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
            prob = self.get_word_probability(word, context)
            if is_shortcut:
                final_score = prob * priority_mult * 1000
            elif is_exact_match:
                final_score = prob * priority_mult * 10
            else:
                final_score = prob * priority_mult
            if is_shortcut and prefix in self.user_shortcut_usage:
                usage_count = self.user_shortcut_usage[prefix]
                final_score *= (1.0 + min(usage_count / 10.0, 2.0))
            scored.append((word, final_score))
        seen = {}
        for word, score in scored:
            if word not in seen or score > seen[word]:
                seen[word] = score
        unique_scored = [(word, score) for word, score in seen.items()]
        unique_scored.sort(key=lambda x: -x[1])
        return [word for word, score in unique_scored[:max_results]]
    
    def get_next_word_suggestions(self, context=None, max_results=6):
        if context is None or len(context) == 0:
            most_common = self.unigrams.most_common(max_results)
            return [word for word, count in most_common]
        elif len(context) == 1:
            prev = context[0].lower()
            prev = self.resolve_shortcut(prev)
            if prev in self.bigrams:
                most_common = self.bigrams[prev].most_common(max_results)
                return [word for word, count in most_common]
            else:
                most_common = self.unigrams.most_common(max_results)
                return [word for word, count in most_common]
        else:
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            trigram_context = (prev2, prev1)
            if trigram_context in self.trigrams:
                most_common = self.trigrams[trigram_context].most_common(max_results)
                return [word for word, count in most_common]
            else:
                if prev1 in self.bigrams:
                    most_common = self.bigrams[prev1].most_common(max_results)
                    return [word for word, count in most_common]
                else:
                    most_common = self.unigrams.most_common(max_results)
                    return [word for word, count in most_common]

# Initialize model
print("="*60)
print("FILIPINO KEYBOARD - LIVE AUTOCOMPLETE VERSION")
print("Gaze-Based Digital Keyboard")
print("="*60)

ngram_model = NgramModel()
if not ngram_model.load_cache():
    print("\nNo cache found. Building from built-in vocabulary...")
    ngram_model.train_from_builtin()
    ngram_model.save_cache()
print("\nLoading user learning...")
ngram_model.load_user_learning()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_current_token(text):
    if not text:
        return ""
    words = text.split()
    if text.endswith(" "):
        return ""
    return words[-1] if words else ""

def get_context_words(text, n=2):
    if not text:
        return []
    words = text.strip().split()
    if not text.endswith(" ") and words:
        words = words[:-1]
    return words[-n:] if len(words) >= n else words

# =============================================================================
# SYNCHRONOUS DWELL HOVER ENGINE
# (based on Algorithm 2 from Meena & Salvi, 2025)
#
# Trial period  ∆t2 = DWELL_TRIAL_MS  (e.g. 1000 ms)
# Poll interval      = DWELL_POLL_MS   (e.g. 50 ms  → 20 frames/trial)
# Weight formula     : W(btn) += √t   per frame while btn is hovered
# Selection          : btn with max W is fired if
#                      P = max(W) / mean(W) ≥ α  (α = 6 → ~60% dominance)
# =============================================================================
DWELL_POLL_MS   = 50     # how often we update (ms)
DWELL_ENABLED   = True   # can be toggled from Settings
DWELL_MIN_MS    = 600    # ms of continuous hover required to fire


# =============================================================================
# GUI WITH LIVE AUTOCOMPLETE
# =============================================================================
class FilipinoKeyboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filipino Keyboard - Live Autocomplete (Gaze-Based)")
        self.attributes('-fullscreen', True)
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))
        self.configure(bg="#f0f0f0")
        
        self.current_completion = ""
        self.alternative_suggestions = []
        self.current_input = ""
        self.output_words = []
        self.output_cursor = -1
        
        self.current_theme = "light"
        self.themes = {
            "light": {
                "bg": "#f0f0f0",
                "output_bg": "#ffffff",
                "input_bg": "#f9f9f9",
                "text_fg": "black",
                "suggestion_fg": "gray",
                "popup_bg": "#fff9e6",
                "popup_border": "#ffcc00",
                "popup_top_bg": "#d4edda",
                "popup_top_fg": "#155724",
                "button_bg": "#e0e0e0",
                "button_fg": "black",
                "button_active_bg": "#d0d0d0",
                "dwell_bar":  "#00cc44",
                "dwell_bg":   "#c8f0d8",
            },
            "dark": {
                "bg": "#36393f",
                "output_bg": "#2f3136",
                "input_bg": "#40444b",
                "text_fg": "#dcddde",
                "suggestion_fg": "#8e9297",
                "popup_bg": "#202225",
                "popup_border": "#5865f2",
                "popup_top_bg": "#1e3a27",
                "popup_top_fg": "#88ffaa",
                "button_bg": "#4f545c",
                "button_fg": "#ffffff",
                "button_active_bg": "#5865f2",
                "dwell_bar":  "#55ff88",
                "dwell_bg":   "#1a3a28",
            }
        }

        # ── Dwell state ───────────────────────────────────────────────────────
        self.dwell_enabled   = DWELL_ENABLED
        self.dwell_hover_ms  = {}      # btn_id → cumulative hover time this dwell (ms)
        self.dwell_hovered   = None    # currently hovered button widget
        self.dwell_trial_job = None    # after() handle for poll loop
        self.dwell_overlays  = {}      # btn_id → Canvas progress bar
        self.dwell_btn_meta  = {}      # btn_id → (widget, command_callable)

        self.alt_popup = None
        self.create_widgets()
    
    def create_widgets(self):
        self.output_display = tk.Text(self, wrap="word", font=("Segoe UI", 18), height=2)
        self.output_display.pack(fill="x", padx=5, pady=(5, 3))
        self.output_display.config(state="disabled")
        
        self.input_display = tk.Text(self, wrap="word", font=("Segoe UI", 16), height=1)
        self.input_display.pack(fill="x", padx=5, pady=3)
        self.input_display.config(state="disabled")
        
        self.predictive_container = ttk.Frame(self)
        self.predictive_container.pack(fill="x", padx=5, pady=3)
        
        keyboard_frame = tk.Frame(self, bg=self.themes[self.current_theme]["bg"])
        keyboard_frame.pack(fill="both", expand=True, padx=5, pady=(3, 5))
        self.create_keyboard(keyboard_frame)
        
        self.status_bar = ttk.Label(
            self,
            text="Type → SPACE to add word → ENTER to speak & clear | ◄► navigate words",
            relief="sunken", anchor="w", font=("Segoe UI", 8)
        )
        self.status_bar.pack(fill="x", side="bottom")
        
        self.apply_theme()
        self.update_display()
        # Start the synchronous dwell trial loop
        self.after(500, self._dwell_start_trial)
    
    # =========================================================================
    # SYNCHRONOUS DWELL ENGINE  (Algorithm 2, Meena & Salvi 2025)
    # =========================================================================
    def _dwell_register(self, btn, command):
        """Register a button for dwell hover interaction."""
        bid = id(btn)
        self.dwell_btn_meta[bid] = (btn, command)
        self.dwell_hover_ms[bid] = 0
        btn.bind("<Enter>", lambda e, b=btn: self._dwell_enter(b))
        btn.bind("<Leave>", lambda e, b=btn: self._dwell_leave(b))
        btn.bind("<Map>",   lambda e, b=btn: self._dwell_create_overlay(b))

    def _dwell_create_overlay(self, btn):
        """Create a slim 6-px Canvas progress bar at the bottom of btn."""
        bid = id(btn)
        if bid in self.dwell_overlays:
            return
        try:
            w = btn.winfo_width()
            if w < 2:
                btn.after(100, lambda: self._dwell_create_overlay(btn))
                return
            theme = self.themes[self.current_theme]
            canvas = tk.Canvas(btn.master, height=6, bd=0,
                                highlightthickness=0, bg=theme["dwell_bg"])
            canvas.place(in_=btn, relx=0, rely=1.0, anchor="sw",
                         relwidth=1.0, height=6)
            canvas.lift()
            canvas.create_rectangle(0, 0, 0, 6,
                                    fill=theme["dwell_bar"],
                                    outline="", tags="bar")
            self.dwell_overlays[bid] = canvas
        except Exception:
            pass

    def _dwell_enter(self, btn):
        """Mouse entered — highlight border green (paper: white→green)."""
        if not self.dwell_enabled:
            return
        self.dwell_hovered = btn
        try:
            btn.config(highlightthickness=3,
                       highlightbackground="#00cc44",
                       highlightcolor="#00cc44")
        except Exception:
            pass

    def _dwell_leave(self, btn):
        """Mouse left — remove highlight and RESET that button's hover counter."""
        if not self.dwell_enabled:
            return
        if self.dwell_hovered is btn:
            self.dwell_hovered = None
        try:
            btn.config(highlightthickness=0)
        except Exception:
            pass
        bid = id(btn)
        # Reset this button's progress — moving away cancels it
        self.dwell_hover_ms[bid] = 0
        if bid in self.dwell_overlays:
            try:
                self.dwell_overlays[bid].coords("bar", 0, 0, 0, 6)
            except Exception:
                pass

    def _dwell_start_trial(self):
        """Start the dwell polling loop."""
        if self.dwell_trial_job:
            self.after_cancel(self.dwell_trial_job)
        self._dwell_reset_all()
        self._dwell_tick()

    def _dwell_reset_all(self):
        """Zero all hover times and progress bars."""
        for bid in self.dwell_hover_ms:
            self.dwell_hover_ms[bid] = 0
        for canvas in self.dwell_overlays.values():
            try:
                canvas.coords("bar", 0, 0, 0, 6)
            except Exception:
                pass

    def _dwell_tick(self):
        """
        Poll every DWELL_POLL_MS.

        Logic:
          - Track cumulative hover ms for the currently hovered button.
          - Progress bar fills from 0 → full as hover_ms approaches DWELL_MIN_MS.
          - As soon as hover_ms >= DWELL_MIN_MS → fire that button immediately.
          - If the mouse leaves before threshold → that button's counter resets
            to 0 (no accidental accumulation across moves).
          - Nothing fires if no key has been held long enough. Period.
        """
        if not self.dwell_enabled:
            self.dwell_trial_job = self.after(DWELL_POLL_MS, self._dwell_tick)
            return

        if self.dwell_hovered is not None:
            bid = id(self.dwell_hovered)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] += DWELL_POLL_MS
                elapsed = self.dwell_hover_ms[bid]

                # Update progress bar (fills toward DWELL_MIN_MS)
                if bid in self.dwell_overlays:
                    canvas = self.dwell_overlays[bid]
                    try:
                        cw = canvas.winfo_width()
                        if cw > 1:
                            progress = min(elapsed / DWELL_MIN_MS, 1.0)
                            canvas.coords("bar", 0, 0, int(cw * progress), 6)
                    except Exception:
                        pass

                # Threshold reached → fire immediately
                if elapsed >= DWELL_MIN_MS:
                    print(f"✓ Dwell fired after {elapsed}ms")
                    self._dwell_fire(bid)
                    # Full reset so same key can't double-fire next tick
                    self._dwell_reset_all()

        self.dwell_trial_job = self.after(DWELL_POLL_MS, self._dwell_tick)

    def _dwell_fire(self, bid):
        """Fire the command for the given button id."""
        meta = self.dwell_btn_meta.get(bid)
        if meta:
            btn, command = meta
            self._dwell_flash(btn)
            try:
                command()
            except Exception as e:
                print(f"⚠ Dwell command error: {e}")

    def _dwell_flash(self, btn):
        """Green flash = confirmation feedback (replaces audio beep)."""
        theme = self.themes[self.current_theme]
        try:
            btn.config(bg="#00cc44", fg="#ffffff")
            btn.after(200, lambda: btn.config(
                bg=theme["button_bg"], fg=theme["button_fg"]))
        except Exception:
            pass

    def _make_dwell_btn(self, parent, command, **kwargs):
        """Create a Button that supports BOTH click and synchronous dwell."""
        btn = tk.Button(parent, command=command, **kwargs)
        self._dwell_register(btn, command)
        return btn

    def update_display(self):
        """Update displays — fetch suggestions first, then draw."""
        theme = self.themes[self.current_theme]
        
        # ── STEP 1: Fetch suggestions ────────────────────────────────────────
        if self.current_input:
            context_words = (
                self.output_words[:self.output_cursor]
                if self.output_cursor != -1
                else self.output_words
            )
            context = get_context_words(" ".join(context_words), n=2)
            suggestions = ngram_model.get_completion_suggestions(
                self.current_input, context, max_results=5
            )
            print(f"🔍 Input: '{self.current_input}' → Suggestions: {suggestions}")
            if suggestions:
                self.current_completion = suggestions[0]
                self.alternative_suggestions = suggestions[1:5]
            else:
                self.current_completion = self.current_input
                self.alternative_suggestions = []
        else:
            self.current_completion = ""
            self.alternative_suggestions = []
        
        # ── STEP 2: Draw output display ──────────────────────────────────────
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        
        for i, word in enumerate(self.output_words):
            if i == self.output_cursor and self.current_input:
                continue  # Skip — replaced by live completion below
            else:
                if i == self.output_cursor:
                    self.output_display.insert("end", word + " ", "highlighted_word")
                else:
                    self.output_display.insert("end", word + " ", "normal")
        
        if self.current_input:
            if self.current_completion and self.current_completion != self.current_input:
                self.output_display.insert("end", self.current_completion, "suggestion")
            else:
                self.output_display.insert("end", self.current_input, "typing")
            self.output_display.insert("end", " |", "cursor")
        else:
            output_text = " ".join(self.output_words)
            if output_text:
                self.output_display.delete("1.0", "end")
                self.output_display.insert("1.0", output_text + " ")
            self.output_display.insert("end", "|", "cursor")
        
        self.output_display.tag_config("cursor", foreground="red", font=("Segoe UI", 18, "bold"))
        self.output_display.tag_config("normal", foreground=theme["text_fg"])
        self.output_display.tag_config("typing", foreground=theme["text_fg"])
        self.output_display.tag_config("suggestion", foreground=theme["suggestion_fg"])
        self.output_display.tag_config(
            "highlighted_word",
            foreground=theme["text_fg"],
            background="#ffe066" if self.current_theme == "light" else "#5865f2"
        )
        self.output_display.config(state="disabled")
        
        # ── STEP 3: Draw input display ───────────────────────────────────────
        self.input_display.config(state="normal")
        self.input_display.delete("1.0", "end")
        
        for i, word in enumerate(self.output_words):
            if i == self.output_cursor and self.current_input:
                self.input_display.insert("end", self.current_input, "editing")
                self.input_display.insert("end", "|", "cursor")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
            elif i == self.output_cursor:
                self.input_display.insert("end", word, "highlighted")
                self.input_display.insert("end", "|", "cursor")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
            else:
                self.input_display.insert("end", word, "normal")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
        
        if self.output_cursor == -1 or self.output_cursor >= len(self.output_words):
            if self.output_words:
                self.input_display.insert("end", " ")
            if self.current_input:
                self.input_display.insert("end", self.current_input, "editing")
                self.input_display.insert("end", "|", "cursor")
            else:
                self.input_display.insert("end", "|", "cursor")
        
        self.input_display.tag_config("cursor", foreground="red", font=("Segoe UI", 16, "bold"))
        self.input_display.tag_config("highlighted", foreground=theme["text_fg"], background="yellow")
        self.input_display.tag_config("editing", foreground=theme["text_fg"])
        self.input_display.tag_config("normal", foreground=theme["text_fg"])
        self.input_display.config(state="disabled")
        
        # ── STEP 4: Popup & predictions ──────────────────────────────────────
        if self.current_input and self.alternative_suggestions:
            self.show_alternative_popup()
        else:
            self.close_popup()
        
        self.update_predictions()
    
    # =========================================================================
    # POPUP — suggestions only, no labels, maximized large buttons
    # =========================================================================
    def show_alternative_popup(self):
        """Show all suggestions as large, tap-friendly buttons. No labels, no clutter."""
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None
        
        if not self.alternative_suggestions:
            return
        
        theme = self.themes[self.current_theme]

        self.alt_popup = tk.Toplevel(self)
        self.alt_popup.overrideredirect(True)
        self.alt_popup.attributes('-topmost', True)

        # Thin colored border as the outermost frame
        border_frame = tk.Frame(self.alt_popup, bg=theme["popup_border"])
        border_frame.pack(fill="both", expand=True)

        # Inner frame — zero padding so buttons fill all available space
        inner_frame = tk.Frame(border_frame, bg=theme["popup_bg"])
        inner_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # All words: top completion first, then alternatives
        all_words = [self.current_completion] + list(self.alternative_suggestions[:4])

        for idx, word in enumerate(all_words):
            is_top = (idx == 0)

            btn = tk.Button(
                inner_frame,
                text=word,
                command=lambda w=word: self.apply_alternative_from_popup(w),
                font=("Segoe UI", 20, "bold") if is_top else ("Segoe UI", 18),
                relief="flat",
                bd=0,
                padx=20,
                pady=18,
                cursor="hand2",
                anchor="center",
                bg=theme["popup_top_bg"] if is_top else theme["popup_bg"],
                fg=theme["popup_top_fg"] if is_top else theme["text_fg"],
                activebackground=theme["button_active_bg"],
                activeforeground=theme["button_fg"],
            )
            btn.pack(side="left", fill="y", padx=0, pady=0)

            # Thin 1 px vertical divider between items (skip after last)
            if idx < len(all_words) - 1:
                divider = tk.Frame(inner_frame, bg=theme["popup_border"], width=1)
                divider.pack(side="left", fill="y")

        # Sizes must be known before positioning
        self.alt_popup.update_idletasks()
        self._position_popup_at_typed_word()

        # Auto-close after 8 seconds
        self.alt_popup.after(8000, self.close_popup)
    
    def _position_popup_at_typed_word(self):
        """
        Position the popup immediately after the typed / completion word
        shown in the output display.
        """
        if not self.alt_popup:
            return
        
        try:
            self.output_display.update_idletasks()

            char_offset = 0
            for i, word in enumerate(self.output_words):
                if i == self.output_cursor and self.current_input:
                    break
                char_offset += len(word) + 1  # +1 for space

            start_index = f"1.{char_offset}"
            end_index   = f"1.{char_offset + len(self.current_completion)}"

            bbox_start = self.output_display.bbox(start_index)
            bbox_end   = self.output_display.bbox(end_index)

            widget_x = self.output_display.winfo_rootx()
            widget_y = self.output_display.winfo_rooty()
            widget_h = self.output_display.winfo_height()

            popup_w = self.alt_popup.winfo_width()
            popup_h = self.alt_popup.winfo_height()
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()

            if bbox_end:
                ex, ey, ew, eh = bbox_end
                x = widget_x + ex + ew + 6
                y = widget_y + ey + eh + 4
            elif bbox_start:
                sx, sy, sw, sh = bbox_start
                x = widget_x + sx
                y = widget_y + sy + sh + 4
            else:
                x = widget_x + 20
                y = widget_y + widget_h + 4

            if x + popup_w > screen_w - 10:
                x = screen_w - popup_w - 10
            if x < 10:
                x = 10
            if y + popup_h > screen_h - 10:
                if bbox_start:
                    sx, sy, sw, sh = bbox_start
                    y = widget_y + sy - popup_h - 4
                else:
                    y = screen_h - popup_h - 10
            if y < 0:
                y = 0

            self.alt_popup.geometry(f"+{x}+{y}")

        except Exception as e:
            print(f"⚠ Popup positioning error: {e}")
            try:
                wx = self.output_display.winfo_rootx() + 20
                wy = self.output_display.winfo_rooty() + self.output_display.winfo_height() + 4
                self.alt_popup.geometry(f"+{wx}+{wy}")
            except Exception:
                pass
    
    def close_popup(self):
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None
    
    # =========================================================================
    # apply_alternative — REPLACE word at cursor instead of inserting
    # =========================================================================
    def apply_alternative_from_popup(self, word):
        """Apply alternative suggestion from popup."""
        self.close_popup()
        self._commit_word(word)
    
    def apply_alternative(self, word):
        """Apply alternative suggestion (called from buttons/other paths)."""
        self.close_popup()
        self._commit_word(word)
    
    def _commit_word(self, word):
        """
        Core logic to finalize a word, correctly REPLACING the word at
        output_cursor when editing, or appending at the end.
        """
        context_words = (
            self.output_words[:self.output_cursor]
            if self.output_cursor != -1
            else self.output_words
        )
        context = get_context_words(" ".join(context_words), n=2)
        
        if self.current_input and self.current_input != word:
            ngram_model.learn_from_user_typing(self.current_input, word)
        ngram_model.track_word_usage(word, context)
        
        if self.output_cursor != -1 and self.output_cursor < len(self.output_words):
            self.output_words[self.output_cursor] = word
            self.output_cursor += 1
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
        else:
            self.output_words.append(word)
            self.output_cursor = -1
        
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.update_display()
        self.status_bar.config(text=f"Selected: '{word}'")
    
    # =========================================================================
    # SPACE — finalize current word
    # =========================================================================
    def finalize_word(self):
        """SPACE — accept the top completion (or typed word) and commit it."""
        if not self.current_input:
            self.status_bar.config(text="Nothing to finalize")
            return
        self.close_popup()
        word_to_add = self.current_completion if self.current_completion else self.current_input
        print(f"\n🔹 FINALIZE: input='{self.current_input}' → '{word_to_add}'")
        self._commit_word(word_to_add)
        self.status_bar.config(text=f"Added '{word_to_add}'")
    
    # =========================================================================
    # NAVIGATION
    # =========================================================================
    def move_word_left(self):
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        if self.output_cursor == -1:
            self.output_cursor = len(self.output_words) - 1
        elif self.output_cursor > 0:
            self.output_cursor -= 1
        self.current_input = ""
        self.update_display()
        self.status_bar.config(text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'")
    
    def move_word_right(self):
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        if self.output_cursor == -1:
            self.status_bar.config(text="Already at end")
            return
        if self.output_cursor < len(self.output_words) - 1:
            self.output_cursor += 1
        else:
            self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        if self.output_cursor == -1:
            self.status_bar.config(text="Cursor at end — ready for new word")
        else:
            self.status_bar.config(text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'")
    
    def clear_selected_word(self):
        if self.output_cursor == -1:
            self.status_bar.config(text="No word selected — cursor at end")
            return
        if self.output_cursor >= len(self.output_words):
            self.status_bar.config(text="No word at cursor")
            return
        removed_word = self.output_words.pop(self.output_cursor)
        if self.output_cursor >= len(self.output_words):
            self.output_cursor = -1
        self.update_display()
        self.status_bar.config(text=f"Removed '{removed_word}'")
    
    # =========================================================================
    # PREDICTIONS
    # =========================================================================
    def update_predictions(self):
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        output_text = " ".join(self.output_words)
        context = get_context_words(output_text, n=2)
        predictions = ngram_model.get_next_word_suggestions(context, max_results=6)
        if predictions:
            for word in predictions[:6]:
                btn = tk.Button(
                    self.predictive_container,
                    text=word,
                    command=lambda w=word: self.apply_prediction(w),
                    font=("Segoe UI", 14, "bold"),
                    relief="raised",
                    bd=2,
                    cursor="hand2"
                )
                btn.pack(side="left", padx=3, ipadx=20, ipady=12, expand=True, fill="both")
    
    def apply_prediction(self, word):
        output_text = " ".join(self.output_words)
        context = get_context_words(output_text, n=2)
        self.output_words.append(word)
        self.output_cursor = -1
        ngram_model.track_word_usage(word, context)
        self.update_display()
        self.status_bar.config(text=f"Predicted: '{word}'")
    
    # =========================================================================
    # INPUT
    # =========================================================================
    def insert_char(self, char):
        self.current_input += char
        self.update_display()
        self.status_bar.config(text=f"Typing: '{self.current_input}'")
    
    def move_cursor_left(self):
        if self.current_input:
            self.current_input = self.current_input[:-1]
            self.update_display()
            self.status_bar.config(text="Removed last character")
        else:
            self.status_bar.config(text="Input is empty")
    
    def move_cursor_right(self):
        if self.current_input:
            self.current_input = ""
            self.current_completion = ""
            self.alternative_suggestions = []
            self.update_display()
            self.status_bar.config(text="Input cleared")
        else:
            self.status_bar.config(text="Input already empty")
    
    def backspace(self):
        print(f"\n⌫ BACKSPACE: input='{self.current_input}' words={self.output_words} cursor={self.output_cursor}")
        if self.current_input:
            self.current_input = self.current_input[:-1]
            self.current_completion = ""
            self.alternative_suggestions = []
            self.update_display()
            self.status_bar.config(text="Backspace")
        elif self.output_cursor == -1 and self.output_words:
            last_word = self.output_words.pop()
            self.current_input = last_word[:-1] if len(last_word) > 1 else ""
            self.update_display()
            self.status_bar.config(text=f"Editing: '{last_word}' → '{self.current_input}'")
        elif self.output_cursor != -1 and self.output_cursor < len(self.output_words):
            deleted_word = self.output_words.pop(self.output_cursor)
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
            self.update_display()
            self.status_bar.config(text=f"Deleted word: '{deleted_word}'")
        else:
            self.status_bar.config(text="Nothing to delete")
    
    def enter(self):
        if self.current_input:
            self.finalize_word()
        output_text = " ".join(self.output_words)
        if output_text.strip():
            print(f"🔊 TTS: {output_text.strip()}")
        self.output_words = []
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        self.status_bar.config(text="Spoken and cleared")
    
    def clear_all(self):
        self.output_words = []
        self.output_cursor = -1
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        self.update_display()
        self.status_bar.config(text="All cleared")
    
    # =========================================================================
    # SETTINGS / THEME
    # =========================================================================
    def show_settings(self):
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("420x380")
        settings_window.resizable(False, False)
        settings_window.transient(self)
        settings_window.grab_set()

        # ── Theme ─────────────────────────────────────────────────────────────
        theme_frame = ttk.LabelFrame(settings_window, text="Theme", padding=12)
        theme_frame.pack(fill="x", padx=20, pady=(15, 8))
        ttk.Label(theme_frame, text="Select Theme:", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 8))
        btn_row = tk.Frame(theme_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="☀ Light Mode",
                   command=lambda: self.change_theme("light", settings_window),
                   width=18).pack(side="left", padx=(0, 10), ipady=8)
        ttk.Button(btn_row, text="🌙 Dark Mode",
                   command=lambda: self.change_theme("dark", settings_window),
                   width=18).pack(side="left", ipady=8)
        ttk.Label(theme_frame, text=f"Current: {self.current_theme.capitalize()} Mode",
                  font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(8, 0))

        # ── Dwell ─────────────────────────────────────────────────────────────
        dwell_frame = ttk.LabelFrame(settings_window, text="Hover Dwell Input", padding=12)
        dwell_frame.pack(fill="x", padx=20, pady=8)

        dwell_var = tk.BooleanVar(value=self.dwell_enabled)
        ttk.Checkbutton(
            dwell_frame, text="Enable hover dwell input",
            variable=dwell_var,
            command=lambda: self._toggle_dwell(dwell_var.get())
        ).pack(anchor="w")

        ttk.Label(dwell_frame,
                  text="Hold duration — how long to hover before key fires:",
                  font=("Segoe UI", 9)).pack(anchor="w", pady=(12, 2))

        min_ms_var = tk.IntVar(value=DWELL_MIN_MS)
        min_row = tk.Frame(dwell_frame)
        min_row.pack(fill="x")
        min_label = ttk.Label(min_row, text=f"{DWELL_MIN_MS} ms", width=7)
        min_label.pack(side="right")

        def on_min_slider(val):
            v = int(float(val) // 50) * 50
            min_label.config(text=f"{v} ms")
            min_ms_var.set(v)

        ttk.Scale(min_row, from_=200, to=1500,
                  orient="horizontal", variable=min_ms_var,
                  command=on_min_slider
                  ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        ttk.Button(
            dwell_frame, text="Apply",
            command=lambda: self._apply_min_hover(min_ms_var.get())
        ).pack(anchor="e", pady=(8, 0))

        ttk.Label(dwell_frame,
                  text="Moving off a key resets its progress to zero.",
                  font=("Segoe UI", 8, "italic"), foreground="gray"
                  ).pack(anchor="w", pady=(6, 0))

        # ── Close ─────────────────────────────────────────────────────────────
        ttk.Button(settings_window, text="Close",
                   command=settings_window.destroy).pack(pady=(8, 12))

    def _toggle_dwell(self, enabled):
        global DWELL_ENABLED
        DWELL_ENABLED = enabled
        self.dwell_enabled = enabled
        state = "ON" if enabled else "OFF"
        self.status_bar.config(text=f"Hover dwell: {state}")

    def _apply_min_hover(self, ms):
        global DWELL_MIN_MS
        DWELL_MIN_MS = max(50, int(ms))
        self.status_bar.config(text=f"Minimum hover time set to {DWELL_MIN_MS} ms")
    
    def change_theme(self, theme, settings_window=None):
        self.current_theme = theme
        self.apply_theme()
        if settings_window:
            settings_window.destroy()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")
    
    def apply_theme(self):
        theme = self.themes[self.current_theme]
        self.configure(bg=theme["bg"])
        self.output_display.config(bg=theme["output_bg"], fg=theme["text_fg"], insertbackground=theme["text_fg"])
        self.input_display.config(bg=theme["input_bg"], fg=theme["text_fg"], insertbackground=theme["text_fg"])
        self.output_display.tag_config("input", foreground=theme["text_fg"])
        self.output_display.tag_config("suggestion", foreground=theme["suggestion_fg"])
        if hasattr(self, 'keyboard_buttons'):
            for btn in self.keyboard_buttons:
                btn.config(
                    bg=theme["button_bg"],
                    fg=theme["button_fg"],
                    activebackground=theme["button_active_bg"],
                    activeforeground=theme["button_fg"]
                )
        if hasattr(self, 'predictive_container'):
            for widget in self.predictive_container.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.config(
                        bg=theme["button_bg"],
                        fg=theme["button_fg"],
                        activebackground=theme["button_active_bg"],
                        activeforeground=theme["button_fg"]
                    )
    
    # =========================================================================
    # KEYBOARD LAYOUT
    # =========================================================================
    def create_keyboard(self, parent):
        self.keyboard_buttons = []
        theme = self.themes[self.current_theme]
        
        main_container = tk.Frame(parent, bg=theme["bg"])
        main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        for i in range(5):
            main_container.grid_rowconfigure(i, weight=1, uniform="row")
        main_container.grid_columnconfigure(0, weight=1)
        
        # Row 0: ◄  CLEAR ALL  SETTINGS  ►
        row0 = tk.Frame(main_container, bg=theme["bg"])
        row0.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        row0.grid_rowconfigure(0, weight=1)
        for col, w in enumerate([1, 3, 3, 1]):
            row0.grid_columnconfigure(col, weight=w, uniform="func")
        
        for col, (text, cmd) in enumerate([
            ("◄", self.move_word_left),
            ("CLEAR ALL", self.clear_all),
            ("⚙ SETTINGS", self.show_settings),
            ("►", self.move_word_right),
        ]):
            btn = self._make_dwell_btn(
                row0, cmd,
                text=text,
                font=("Segoe UI", 20 if col in (0, 3) else 16, "bold"),
                relief="raised", bd=1, cursor="hand2"
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=0, pady=0)
            self.keyboard_buttons.append(btn)
        
        # Row 1: Q–P
        row1 = tk.Frame(main_container, bg=theme["bg"])
        row1.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        row1.grid_rowconfigure(0, weight=1)
        for i, ch in enumerate("qwertyuiop"):
            row1.grid_columnconfigure(i, weight=1, uniform="key")
            btn = self._make_dwell_btn(
                row1, lambda c=ch: self.insert_char(c),
                text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                relief="raised", bd=1, cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew")
            self.keyboard_buttons.append(btn)
        
        # Row 2: A–L
        row2 = tk.Frame(main_container, bg=theme["bg"])
        row2.grid(row=2, column=0, sticky="nsew", padx=1, pady=1)
        row2.grid_rowconfigure(0, weight=1)
        for i, ch in enumerate("asdfghjkl"):
            row2.grid_columnconfigure(i, weight=1, uniform="key")
            btn = self._make_dwell_btn(
                row2, lambda c=ch: self.insert_char(c),
                text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                relief="raised", bd=1, cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew")
            self.keyboard_buttons.append(btn)
        
        # Row 3: Z–M + ⌫
        row3 = tk.Frame(main_container, bg=theme["bg"])
        row3.grid(row=3, column=0, sticky="nsew", padx=1, pady=1)
        row3.grid_rowconfigure(0, weight=1)
        for i, ch in enumerate("zxcvbnm"):
            row3.grid_columnconfigure(i, weight=1, uniform="key")
            btn = self._make_dwell_btn(
                row3, lambda c=ch: self.insert_char(c),
                text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                relief="raised", bd=1, cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew")
            self.keyboard_buttons.append(btn)
        row3.grid_columnconfigure(7, weight=2, uniform="key")
        bs_btn = self._make_dwell_btn(
            row3, self.backspace,
            text="⌫", font=("Segoe UI", 26, "bold"),
            relief="raised", bd=1, cursor="hand2"
        )
        bs_btn.grid(row=0, column=7, sticky="nsew")
        self.keyboard_buttons.append(bs_btn)
        
        # Row 4: SPACE + ↵
        row4 = tk.Frame(main_container, bg=theme["bg"])
        row4.grid(row=4, column=0, sticky="nsew", padx=1, pady=1)
        row4.grid_rowconfigure(0, weight=1)
        row4.grid_columnconfigure(0, weight=85, uniform="bottom")
        row4.grid_columnconfigure(1, weight=15, uniform="bottom")
        
        sp_btn = self._make_dwell_btn(
            row4, self.finalize_word,
            text="SPACE", font=("Segoe UI", 22, "bold"),
            relief="raised", bd=1, cursor="hand2"
        )
        sp_btn.grid(row=0, column=0, sticky="nsew")
        self.keyboard_buttons.append(sp_btn)
        
        en_btn = self._make_dwell_btn(
            row4, self.enter,
            text="↵", font=("Segoe UI", 26, "bold"),
            relief="raised", bd=1, cursor="hand2"
        )
        en_btn.grid(row=0, column=1, sticky="nsew")
        self.keyboard_buttons.append(en_btn)


if __name__ == "__main__":
    app = FilipinoKeyboard()
    app.mainloop()