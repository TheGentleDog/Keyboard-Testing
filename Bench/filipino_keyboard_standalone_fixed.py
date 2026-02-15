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
    """
    Calculate Damerau-Levenshtein distance between two strings.
    Includes: insertions, deletions, substitutions, and transpositions.
    """
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
        """Train from built-in Filipino vocabulary"""
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
        
        print(f"✓ Built-in words: {len(FILIPINO_WORDS)}")
        print(f"✓ Communication phrases: {len(COMMUNICATION_CORPUS)}")
        print(f"✓ Built-in shortcuts: {len(FILIPINO_SHORTCUTS)}")
        print(f"✓ Total tokens for training: {len(all_tokens)}")
        
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
        
        print(f"\n✓ Vocabulary: {len(self.vocabulary)} unique words")
        print(f"✓ Total words: {self.total_words}")
        print(f"✓ Bigrams: {len(self.bigrams)}")
        print(f"✓ Trigrams: {len(self.trigrams)}")
        print(f"✓ Shortcuts loaded: {len(self.csv_shortcuts)}")
    
    def _has_vowels(self, word):
        """Check if word has vowels (filter acronyms)"""
        return any(c in 'aeiouAEIOU' for c in word)
    
    def _build_char_ngrams(self, word):
        """Build character-level bigrams within word"""
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
        """Get completions based on character patterns"""
        if len(prefix) < 1:
            return []
        
        if len(prefix) == 1:
            min_length = 2
        else:
            min_length = max(len(prefix) + 1, 3)
        
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
        """Resolve shortcut to full word"""
        word_lower = word.lower()
        
        if word_lower in self.user_shortcuts:
            return self.user_shortcuts[word_lower]
        
        if word_lower in self.csv_shortcuts:
            return self.csv_shortcuts[word_lower]
        
        return word
    
    def get_all_shortcut_expansions(self, shortcut):
        """Get all possible expansions of a shortcut"""
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
        """Learn shortcut from user behavior"""
        typed_shortcut = typed_shortcut.lower()
        selected_word = selected_word.lower()
        
        if len(typed_shortcut) < len(selected_word) - 1:
            if typed_shortcut not in self.user_shortcuts:
                self.user_shortcuts[typed_shortcut] = selected_word
                print(f"🎓 Learned: '{typed_shortcut}' → '{selected_word}'")
            
            self.user_shortcut_usage[typed_shortcut] += 1
            self.save_user_learning()
    
    def add_new_word(self, word):
        """Add new word to vocabulary"""
        word_lower = word.lower()
        if word_lower not in self.vocabulary:
            self.vocabulary.add(word_lower)
            self.new_words.add(word_lower)
            self.unigrams[word_lower] = 1
            print(f"📝 New word: '{word}'")
            self.save_user_learning()
    
    def track_word_usage(self, word, context=None):
        """Track word usage to improve predictions"""
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
        """Save user learning to JSON"""
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
        """Load user learning from JSON"""
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
        """Save model to cache"""
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
        """Load model from cache"""
        if not os.path.exists(NGRAM_CACHE_FILE):
            return False
        
        try:
            with open(NGRAM_CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            
            cached_version = data.get('version', '1.0')
            if cached_version != MODEL_VERSION:
                print(f"⚠ Cache version mismatch ({cached_version} vs {MODEL_VERSION})")
                print(f"⚠ Rebuilding model with new data...")
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
            
            print(f"✓ Loaded cache")
            print(f"  Vocabulary: {len(self.vocabulary)} words")
            print(f"  Shortcuts: {len(self.csv_shortcuts)}")
            return True
        except Exception as e:
            print(f"⚠ Error: {e}")
            return False
    
    def get_word_probability(self, word, context=None):
        """Calculate word probability using n-grams"""
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
        """Get word completion suggestions using ngram and Damerau-Levenshtein"""
        prefix = prefix.lower()
        
        # PRIORITY 1: Shortcuts
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if all_expansions:
            for full_word, source in all_expansions:
                priority_multiplier = 10.0 if source == 'user' else 8.0
                shortcut_candidates.append((full_word, False, True, priority_multiplier))
        
        # PRIORITY 2: Exact prefix matches
        candidates = []
        
        if len(prefix) == 1:
            min_word_length = 2
        else:
            min_word_length = max(len(prefix) + 1, 3)
        
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
        
        # PRIORITY 3: Fuzzy matches using Damerau-Levenshtein
        fuzzy_candidates = []
        if not shortcut_candidates and len(exact_matches) < 3 and len(prefix) >= 2:
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
                    fuzzy_candidates.append((word, False, False, 0.3 / (position + 1)))
                    continue
                
                if abs(len(word) - len(prefix)) <= 2:
                    distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                    if distance <= 1:
                        fuzzy_candidates.append((word, False, False, 0.1))
        
        # Combine all candidates
        all_candidates = shortcut_candidates + candidates + fuzzy_candidates
        
        # Score using ngram probabilities
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
            prob = self.get_word_probability(word, context)
            
            if is_shortcut:
                final_score = prob * priority_mult * 100
            elif is_exact_match:
                final_score = prob * priority_mult * 10
            else:
                final_score = prob * priority_mult
            
            if is_shortcut and prefix in self.user_shortcut_usage:
                usage_count = self.user_shortcut_usage[prefix]
                final_score *= (1.0 + min(usage_count / 10.0, 2.0))
            
            scored.append((word, final_score))
        
        # Remove duplicates
        seen = {}
        for word, score in scored:
            if word not in seen or score > seen[word]:
                seen[word] = score
        
        unique_scored = [(word, score) for word, score in seen.items()]
        unique_scored.sort(key=lambda x: -x[1])
        
        return [word for word, score in unique_scored[:max_results]]
    
    def get_next_word_suggestions(self, context=None, max_results=6):
        """Predict next word using ngrams"""
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
    """Get the word being typed"""
    if not text:
        return ""
    words = text.split()
    if text.endswith(" "):
        return ""
    return words[-1] if words else ""

def get_context_words(text, n=2):
    """Get last n complete words for context"""
    if not text:
        return []
    words = text.strip().split()
    if not text.endswith(" ") and words:
        words = words[:-1]
    return words[-n:] if len(words) >= n else words

# =============================================================================
# GUI WITH LIVE AUTOCOMPLETE
# =============================================================================
class FilipinoKeyboard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Filipino Keyboard - Live Autocomplete (Gaze-Based)")
        self.geometry("1400x1000")  # Taller window for bigger keys
        self.configure(bg="#f0f0f0")
        
        self.current_completion = ""
        self.alternative_suggestions = []
        self.current_input = ""
        self.finalized_text = ""  # Stores the actual finalized output
        
        # Theme settings
        self.current_theme = "light"  # default theme
        self.themes = {
            "light": {
                "bg": "#f0f0f0",
                "output_bg": "#ffffff",
                "input_bg": "#f9f9f9",
                "text_fg": "black",
                "suggestion_fg": "gray",
                "popup_bg": "#fff9e6",
                "popup_border": "#ffcc00",
                "button_bg": "#e0e0e0",
                "button_fg": "black",
                "button_active_bg": "#d0d0d0"
            },
            "dark": {
                "bg": "#36393f",  # Discord dark gray
                "output_bg": "#2f3136",  # Discord darker gray
                "input_bg": "#40444b",  # Discord input gray
                "text_fg": "#dcddde",  # Discord light text
                "suggestion_fg": "#8e9297",  # Discord muted text
                "popup_bg": "#202225",  # Discord darkest
                "popup_border": "#5865f2",  # Discord blurple
                "button_bg": "#4f545c",  # Discord button gray
                "button_fg": "#ffffff",  # White text
                "button_active_bg": "#5865f2"  # Discord blurple on hover
            }
        }
        
        # Alternative popup window
        self.alt_popup = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Output display (top) - VERY COMPACT
        output_frame = ttk.LabelFrame(self, text="Output", padding="3")
        output_frame.pack(fill="x", padx=5, pady=(5, 2))
        
        self.output_display = tk.Text(output_frame, wrap="word", font=("Segoe UI", 14), height=2)
        self.output_display.pack(fill="x")
        self.output_display.config(state="disabled")
        
        # Input display - VERY COMPACT
        input_frame = ttk.LabelFrame(self, text="Current Word", padding="3")
        input_frame.pack(fill="x", padx=5, pady=2)
        
        self.input_display = tk.Text(input_frame, wrap="word", font=("Segoe UI", 12), height=1)
        self.input_display.pack(fill="x")
        self.input_display.config(state="disabled")
        
        # Next word predictions - VERY COMPACT
        predictive_frame = ttk.LabelFrame(self, text="Next Word", padding="3")
        predictive_frame.pack(fill="x", padx=5, pady=2)
        
        self.predictive_container = ttk.Frame(predictive_frame)
        self.predictive_container.pack(fill="x", pady=2)
        
        # Virtual keyboard - MAXIMIZED (takes ALL remaining space, NO padding)
        keyboard_frame = ttk.LabelFrame(self, text="Keyboard", padding="0")
        keyboard_frame.pack(fill="both", expand=True, padx=5, pady=(2, 5))
        
        self.create_keyboard(keyboard_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Type to autocomplete | SPACE to finalize", relief="sunken", anchor="w", font=("Segoe UI", 8))
        self.status_bar.pack(fill="x", side="bottom")
        
        # Apply initial theme
        self.apply_theme()
        
        self.update_display()
    
    def update_display(self):
        """Update all displays with live autocomplete"""
        # Get finalized text (everything except what we're currently typing)
        finalized_text = self.get_finalized_text()
        context = get_context_words(finalized_text, n=2)
        
        # Update input display
        self.input_display.config(state="normal")
        self.input_display.delete("1.0", "end")
        self.input_display.insert("1.0", self.current_input)
        self.input_display.config(state="disabled")
        
        # Get autocomplete suggestions
        if self.current_input:
            suggestions = ngram_model.get_completion_suggestions(
                self.current_input, context, max_results=5
            )
            
            if suggestions:
                self.current_completion = suggestions[0]
                self.alternative_suggestions = suggestions[1:5]
            else:
                self.current_completion = self.current_input
                self.alternative_suggestions = []
            
            # Show live autocomplete in output
            self.output_display.config(state="normal")
            self.output_display.delete("1.0", "end")
            
            # Show finalized text
            if finalized_text:
                self.output_display.insert("1.0", finalized_text)
            
            theme = self.themes[self.current_theme]
            
            # Show current input in normal color
            self.output_display.insert("end", self.current_input, "input")
            
            # Show completion suggestion in gray
            if len(self.current_completion) > len(self.current_input):
                remaining = self.current_completion[len(self.current_input):]
                self.output_display.insert("end", remaining, "suggestion")
            
            # Configure tags
            self.output_display.tag_config("input", foreground=theme["text_fg"])
            self.output_display.tag_config("suggestion", foreground=theme["suggestion_fg"])
            
            self.output_display.config(state="disabled")
            
            # Show popup for alternatives if they exist
            if self.alternative_suggestions:
                self.show_alternative_popup()
            else:
                self.close_popup()
            
        else:
            self.current_completion = ""
            self.alternative_suggestions = []
            
            # Just show finalized text
            self.output_display.config(state="normal")
            self.output_display.delete("1.0", "end")
            if finalized_text:
                self.output_display.insert("1.0", finalized_text)
            self.output_display.config(state="disabled")
            
            # Close popup
            self.close_popup()
        
        # Update next word predictions
        self.update_predictions()
    
    def show_settings(self):
        """Show settings dialog for theme selection"""
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("400x250")
        settings_window.resizable(False, False)
        
        # Center the window
        settings_window.transient(self)
        settings_window.grab_set()
        
        # Theme selection
        theme_frame = ttk.LabelFrame(settings_window, text="Theme", padding=20)
        theme_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ttk.Label(theme_frame, text="Select Theme:", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 15))
        
        # Light mode button
        light_btn = ttk.Button(
            theme_frame, 
            text="☀ Light Mode",
            command=lambda: self.change_theme("light", settings_window),
            width=20
        )
        light_btn.pack(pady=8, ipady=10)
        
        # Dark mode button
        dark_btn = ttk.Button(
            theme_frame, 
            text="🌙 Dark Mode",
            command=lambda: self.change_theme("dark", settings_window),
            width=20
        )
        dark_btn.pack(pady=8, ipady=10)
        
        # Current theme indicator
        current_label = ttk.Label(
            theme_frame, 
            text=f"Current: {self.current_theme.capitalize()} Mode",
            font=("Segoe UI", 9, "italic")
        )
        current_label.pack(pady=(15, 0))
        
        # Close button
        close_btn = ttk.Button(settings_window, text="Close", command=settings_window.destroy)
        close_btn.pack(pady=(0, 10))
    
    def change_theme(self, theme, settings_window=None):
        """Change the application theme"""
        self.current_theme = theme
        self.apply_theme()
        if settings_window:
            settings_window.destroy()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")
    
    def apply_theme(self):
        """Apply the selected theme to all widgets"""
        theme = self.themes[self.current_theme]
        
        # Main window
        self.configure(bg=theme["bg"])
        
        # Output display
        self.output_display.config(
            bg=theme["output_bg"],
            fg=theme["text_fg"],
            insertbackground=theme["text_fg"]
        )
        
        # Input display
        self.input_display.config(
            bg=theme["input_bg"],
            fg=theme["text_fg"],
            insertbackground=theme["text_fg"]
        )
        
        # Update text tags for autocomplete
        self.output_display.tag_config("input", foreground=theme["text_fg"])
        self.output_display.tag_config("suggestion", foreground=theme["suggestion_fg"])
        
        # Update keyboard buttons
        if hasattr(self, 'keyboard_buttons'):
            for btn in self.keyboard_buttons:
                btn.config(
                    bg=theme["button_bg"],
                    fg=theme["button_fg"],
                    activebackground=theme["button_active_bg"],
                    activeforeground=theme["button_fg"]
                )
    
    def get_finalized_text(self):
        """Get the finalized text from output (without current typing preview)"""
        if not hasattr(self, 'finalized_text'):
            self.finalized_text = ""
        return self.finalized_text
    
    def show_alternative_popup(self):
        """Show alternative suggestions as a popup near the predicted word"""
        # Close existing popup if any
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None
        
        if not self.alternative_suggestions:
            return
        
        # Create popup window
        self.alt_popup = tk.Toplevel(self)
        self.alt_popup.overrideredirect(True)  # Remove window decorations
        
        theme = self.themes[self.current_theme]
        
        # Create frame with border
        popup_frame = tk.Frame(
            self.alt_popup, 
            bg=theme["popup_border"], 
            highlightthickness=2,
            highlightbackground=theme["popup_border"]
        )
        popup_frame.pack(fill="both", expand=True)
        
        inner_frame = tk.Frame(popup_frame, bg=theme["popup_bg"], padx=10, pady=8)
        inner_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Label
        label = tk.Label(
            inner_frame, 
            text="Or do you mean:",
            font=("Segoe UI", 9, "italic"),
            bg=theme["popup_bg"],
            fg=theme["text_fg"]
        )
        label.pack(anchor="w")
        
        # Alternative buttons
        btn_frame = tk.Frame(inner_frame, bg=theme["popup_bg"])
        btn_frame.pack(fill="x", pady=(5, 0))
        
        for word in self.alternative_suggestions[:4]:  # Show max 4 alternatives
            btn = tk.Button(
                btn_frame,
                text=word,
                command=lambda w=word: self.apply_alternative_from_popup(w),
                font=("Segoe UI", 10),
                relief="raised",
                bd=1,
                padx=8,
                pady=4,
                cursor="hand2"
            )
            btn.pack(side="left", padx=3)
        
        # Position the popup
        self.position_popup()
        
        # Auto-close popup after 8 seconds
        self.alt_popup.after(8000, self.close_popup)
    
    def position_popup(self):
        """Position popup near the predicted word (follows cursor, doesn't block text)"""
        if not self.alt_popup:
            return
        
        # Update to get accurate sizes
        self.alt_popup.update_idletasks()
        
        # Get the position of the cursor in the output display
        try:
            # Get the index of the end of text
            cursor_index = self.output_display.index("end-1c")
            
            # Get the bounding box of the cursor position
            bbox = self.output_display.bbox(cursor_index)
            
            if bbox:
                # bbox returns (x, y, width, height) relative to the text widget
                cursor_x, cursor_y, cursor_width, cursor_height = bbox
                
                # Convert to screen coordinates
                widget_x = self.output_display.winfo_rootx()
                widget_y = self.output_display.winfo_rooty()
                
                # Get popup size
                popup_width = self.alt_popup.winfo_width()
                popup_height = self.alt_popup.winfo_height()
                
                # Get screen width
                screen_width = self.winfo_screenwidth()
                
                # Position below the cursor line to not block text
                x = widget_x + cursor_x + cursor_width + 10  # 10px offset from cursor
                y = widget_y + cursor_y + cursor_height + 5   # Below the line
                
                # If popup goes off right edge, move to left of cursor
                if x + popup_width > screen_width - 20:
                    x = widget_x + cursor_x - popup_width - 10
                
                # If still off screen (very left), align with widget
                if x < 20:
                    x = widget_x + 20
                
                self.alt_popup.geometry(f"+{x}+{y}")
            else:
                # Fallback if bbox fails - position at bottom right of output
                output_x = self.output_display.winfo_rootx()
                output_y = self.output_display.winfo_rooty()
                output_height = self.output_display.winfo_height()
                
                x = output_x + 20
                y = output_y + output_height - 100
                
                self.alt_popup.geometry(f"+{x}+{y}")
                
        except Exception as e:
            # Fallback positioning
            output_x = self.output_display.winfo_rootx()
            output_y = self.output_display.winfo_rooty()
            output_height = self.output_display.winfo_height()
            
            x = output_x + 20
            y = output_y + output_height - 100
            
            self.alt_popup.geometry(f"+{x}+{y}")
    
    def close_popup(self):
        """Close the alternative suggestions popup"""
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None
    
    def apply_alternative_from_popup(self, word):
        """Apply alternative suggestion from popup"""
        self.close_popup()
        self.apply_alternative(word)
    
    def update_predictions(self):
        """Update next word prediction buttons"""
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        # Use finalized text for context, not the display with preview
        context = get_context_words(self.finalized_text, n=2)
        
        predictions = ngram_model.get_next_word_suggestions(context, max_results=6)
        
        if predictions:
            for word in predictions[:6]:
                btn = ttk.Button(
                    self.predictive_container,
                    text=word,
                    command=lambda w=word: self.apply_prediction(w),
                    style="Suggestion.TButton"
                )
                btn.pack(side="left", padx=5, ipadx=15, ipady=10)
    
    def finalize_word(self):
        """Finalize current word with SPACE - accepts autocomplete"""
        # Close popup
        self.close_popup()
        
        if not self.current_completion:
            # Just add space if no input
            self.finalized_text += " "
            self.output_display.config(state="normal")
            self.output_display.delete("1.0", "end")
            self.output_display.insert("1.0", self.finalized_text)
            self.output_display.config(state="disabled")
            self.status_bar.config(text="Space added")
            return
        
        # Get context before finalizing
        context = get_context_words(self.finalized_text, n=2)
        
        # Learn from user typing
        if self.current_input != self.current_completion:
            ngram_model.learn_from_user_typing(self.current_input, self.current_completion)
        
        # Track word usage
        ngram_model.track_word_usage(self.current_completion, context)
        
        # Add to finalized text
        self.finalized_text += self.current_completion + " "
        
        # Update output display
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.insert("1.0", self.finalized_text)
        self.output_display.config(state="disabled")
        
        # Clear input
        saved_completion = self.current_completion
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.status_bar.config(text=f"Finalized: '{saved_completion}'")
        self.update_display()
    
    def apply_alternative(self, word):
        """Apply alternative suggestion (when autocomplete gives error)"""
        # Close popup
        self.close_popup()
        
        context = get_context_words(self.finalized_text, n=2)
        
        # Learn and track
        if self.current_input != word:
            ngram_model.learn_from_user_typing(self.current_input, word)
        ngram_model.track_word_usage(word, context)
        
        # Add to finalized text
        self.finalized_text += word + " "
        
        # Update output
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.insert("1.0", self.finalized_text)
        self.output_display.config(state="disabled")
        
        # Clear
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.status_bar.config(text=f"Alternative selected: '{word}'")
        self.update_display()
    
    def apply_prediction(self, word):
        """Apply next word prediction"""
        context = get_context_words(self.finalized_text, n=2)
        
        # Add space if needed before prediction
        if self.finalized_text and not self.finalized_text.endswith(" "):
            self.finalized_text += " "
        
        # Add predicted word
        self.finalized_text += word + " "
        
        # Update output
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.insert("1.0", self.finalized_text)
        self.output_display.config(state="disabled")
        
        ngram_model.track_word_usage(word, context)
        
        self.status_bar.config(text=f"Predicted: '{word}'")
        self.update_display()
    
    def insert_char(self, char):
        """Insert character into current input"""
        self.current_input += char
        self.update_display()
        self.status_bar.config(text=f"Typing: '{self.current_input}'")
    
    def backspace(self):
        """Remove last character from input"""
        if self.current_input:
            # Remove from current input
            self.current_input = self.current_input[:-1]
            self.update_display()
            self.status_bar.config(text="Backspace")
        else:
            # Remove from finalized text
            if self.finalized_text:
                self.finalized_text = self.finalized_text[:-1]
                self.output_display.config(state="normal")
                self.output_display.delete("1.0", "end")
                self.output_display.insert("1.0", self.finalized_text)
                self.output_display.config(state="disabled")
                self.update_display()
    
    def enter(self):
        """New line"""
        # Finalize current word first if exists
        if self.current_input:
            self.finalize_word()
        
        # Add newline to finalized text
        self.finalized_text += "\n"
        
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.insert("1.0", self.finalized_text)
        self.output_display.config(state="disabled")
        self.status_bar.config(text="New line")
        self.update_display()
    
    def clear_all(self):
        """Clear everything"""
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.config(state="disabled")
        
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        self.finalized_text = ""  # Clear finalized text too
        
        self.update_display()
        self.status_bar.config(text="Cleared")
    
    def create_keyboard(self, parent):
        """Create gaze-based virtual keyboard with MAXIMUM space (biggest possible keys)"""
        # Store references for theme updates
        self.keyboard_buttons = []
        
        theme = self.themes[self.current_theme]
        
        # Main container - fills entire parent with NO gaps
        main_container = tk.Frame(parent, bg=theme["bg"])
        main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Configure grid to distribute space evenly with uniform sizing
        for i in range(5):  # 5 rows
            main_container.grid_rowconfigure(i, weight=1, uniform="row")
        main_container.grid_columnconfigure(0, weight=1)
        
        # Row 0: CLEAR ALL and SETTINGS (50/50 split)
        row0_frame = tk.Frame(main_container, bg=theme["bg"])
        row0_frame.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        row0_frame.grid_columnconfigure(0, weight=1, uniform="func")
        row0_frame.grid_columnconfigure(1, weight=1, uniform="func")
        row0_frame.grid_rowconfigure(0, weight=1)
        
        clear_btn = tk.Button(
            row0_frame, text="CLEAR ALL",
            font=("Segoe UI", 18, "bold"),
            command=self.clear_all,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        clear_btn.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(clear_btn)
        
        settings_btn = tk.Button(
            row0_frame, text="⚙ SETTINGS",
            font=("Segoe UI", 18, "bold"),
            command=self.show_settings,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        settings_btn.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(settings_btn)
        
        # Row 1: Q W E R T Y U I O P (10 keys, equal width)
        row1_frame = tk.Frame(main_container, bg=theme["bg"])
        row1_frame.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        row1_frame.grid_rowconfigure(0, weight=1)
        
        keys_row1 = "qwertyuiop"
        for i, ch in enumerate(keys_row1):
            row1_frame.grid_columnconfigure(i, weight=1, uniform="key")
            btn = tk.Button(
                row1_frame, text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                command=lambda c=ch: self.insert_char(c),
                relief="raised",
                bd=1,
                cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew", padx=0, pady=0)
            self.keyboard_buttons.append(btn)
        
        # Row 2: A S D F G H J K L (9 keys, equal width)
        row2_frame = tk.Frame(main_container, bg=theme["bg"])
        row2_frame.grid(row=2, column=0, sticky="nsew", padx=1, pady=1)
        row2_frame.grid_rowconfigure(0, weight=1)
        
        keys_row2 = "asdfghjkl"
        for i, ch in enumerate(keys_row2):
            row2_frame.grid_columnconfigure(i, weight=1, uniform="key")
            btn = tk.Button(
                row2_frame, text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                command=lambda c=ch: self.insert_char(c),
                relief="raised",
                bd=1,
                cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew", padx=0, pady=0)
            self.keyboard_buttons.append(btn)
        
        # Row 3: Z X C V B N M BACKSPACE (8 keys)
        row3_frame = tk.Frame(main_container, bg=theme["bg"])
        row3_frame.grid(row=3, column=0, sticky="nsew", padx=1, pady=1)
        row3_frame.grid_rowconfigure(0, weight=1)
        
        keys_row3 = "zxcvbnm"
        for i, ch in enumerate(keys_row3):
            row3_frame.grid_columnconfigure(i, weight=1, uniform="key")
            btn = tk.Button(
                row3_frame, text=ch.upper(),
                font=("Segoe UI", 22, "bold"),
                command=lambda c=ch: self.insert_char(c),
                relief="raised",
                bd=1,
                cursor="hand2"
            )
            btn.grid(row=0, column=i, sticky="nsew", padx=0, pady=0)
            self.keyboard_buttons.append(btn)
        
        # BACKSPACE button (takes 2 columns worth of space)
        row3_frame.grid_columnconfigure(7, weight=2, uniform="key")
        backspace_btn = tk.Button(
            row3_frame, text="⌫",
            font=("Segoe UI", 26, "bold"),
            command=self.backspace,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        backspace_btn.grid(row=0, column=7, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(backspace_btn)
        
        # Row 4: SPACE (big) and ENTER
        row4_frame = tk.Frame(main_container, bg=theme["bg"])
        row4_frame.grid(row=4, column=0, sticky="nsew", padx=1, pady=1)
        row4_frame.grid_rowconfigure(0, weight=1)
        
        # SPACE takes 85% of width
        row4_frame.grid_columnconfigure(0, weight=85, uniform="bottom")
        space_btn = tk.Button(
            row4_frame, text="SPACE",
            font=("Segoe UI", 22, "bold"),
            command=self.finalize_word,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        space_btn.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(space_btn)
        
        # ENTER takes 15% of width
        row4_frame.grid_columnconfigure(1, weight=15, uniform="bottom")
        enter_btn = tk.Button(
            row4_frame, text="↵",
            font=("Segoe UI", 26, "bold"),
            command=self.enter,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        enter_btn.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(enter_btn)

if __name__ == "__main__":
    app = FilipinoKeyboard()
    app.mainloop()