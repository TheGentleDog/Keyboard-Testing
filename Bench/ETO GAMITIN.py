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
        
        # CRITICAL: Add essential shortcuts that MUST exist (even if not in dataset)
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
            "tyo": "tayo", "kyo": "kayo", "kmi": "kami", "cla": "sila",  # Added pronouns
            "ikw": "ikaw", "ako": "ako"
        }
        
        print(f"📌 Adding {len(essential_shortcuts)} essential shortcuts...")
        for shortcut, full_word in essential_shortcuts.items():
            self.csv_shortcuts[shortcut] = full_word
            all_tokens.extend([full_word.lower()] * 5)
        
        # VERIFY critical shortcuts are loaded
        print(f"\n🔍 VERIFYING SHORTCUTS:")
        critical_shortcuts = ["tyo", "nlng", "kyo", "lng"]
        for sc in critical_shortcuts:
            if sc in self.csv_shortcuts:
                print(f"   ✓ '{sc}' → '{self.csv_shortcuts[sc]}'")
            else:
                print(f"   ✗ '{sc}' MISSING!")
        
        print(f"\n✓ Built-in words: {len(FILIPINO_WORDS)}")
        print(f"✓ Communication phrases: {len(COMMUNICATION_CORPUS)}")
        print(f"✓ Loaded shortcuts: {len(FILIPINO_SHORTCUTS)}")
        print(f"✓ Total shortcuts (with essentials): {len(self.csv_shortcuts)}")
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
        
        # DEBUG
        if prefix == "tyo":
            print(f"\n🔍 DEBUG get_completion_suggestions for 'tyo':")
            print(f"   csv_shortcuts has 'tyo'? {'tyo' in self.csv_shortcuts}")
            if 'tyo' in self.csv_shortcuts:
                print(f"   'tyo' maps to: '{self.csv_shortcuts['tyo']}'")
        
        # PRIORITY 1: Shortcuts
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if prefix == "tyo":
            print(f"   all_expansions: {all_expansions}")
        
        if all_expansions:
            for full_word, source in all_expansions:
                priority_multiplier = 10.0 if source == 'user' else 8.0
                shortcut_candidates.append((full_word, False, True, priority_multiplier))
        
        if prefix == "tyo":
            print(f"   shortcut_candidates: {shortcut_candidates}")
        
        # PRIORITY 2: Exact prefix matches
        candidates = []
        
        # FIXED: Allow minimum 2-letter words to find closer matches
        if len(prefix) == 1:
            min_word_length = 2
        else:
            min_word_length = 2  # Was max(len(prefix) + 1, 3)
        
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
        
        # PRIORITY 3: Fuzzy matches using Damerau-Levenshtein (ALWAYS run to fill suggestions)
        fuzzy_candidates = []
        if len(prefix) >= 2:  # FIXED: Always run fuzzy matching
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
                
                # Only very close matches: distance = 1 AND similar start
                if abs(len(word) - len(prefix)) <= 2:
                    distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                    # ONLY distance 1 for fuzzy (very strict)
                    if distance == 1:
                        similarity = 1.0 / (1.0 + distance)
                        fuzzy_candidates.append((word, False, False, similarity * 0.2))
        
        # Combine all candidates
        all_candidates = shortcut_candidates + candidates + fuzzy_candidates
        
        # Score using ngram probabilities
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
            prob = self.get_word_probability(word, context)
            
            if is_shortcut:
                final_score = prob * priority_mult * 1000  # FIXED: Was 100, now 1000
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
        
        # Return all suggestions (including multi-word phrases)
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
        
        # Make window truly fullscreen (no taskbar)
        self.attributes('-fullscreen', True)
        
        # Bind ESC key to exit fullscreen
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))
        
        self.configure(bg="#f0f0f0")
        
        self.current_completion = ""
        self.alternative_suggestions = []
        self.current_input = ""
        
        # Output word list and cursor
        self.output_words = []  # List of finalized words in output
        self.output_cursor = -1  # Position in output (-1 = at end, typing new word)
        
        # Theme settings
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
                "button_bg": "#e0e0e0",
                "button_fg": "black",
                "button_active_bg": "#d0d0d0"
            },
            "dark": {
                "bg": "#36393f",
                "output_bg": "#2f3136",
                "input_bg": "#40444b",
                "text_fg": "#dcddde",
                "suggestion_fg": "#8e9297",
                "popup_bg": "#202225",
                "popup_border": "#5865f2",
                "button_bg": "#4f545c",
                "button_fg": "#ffffff",
                "button_active_bg": "#5865f2"
            }
        }
        
        # Alternative popup window
        self.alt_popup = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Output display (top) - NO LABEL, just the text box
        self.output_display = tk.Text(self, wrap="word", font=("Segoe UI", 18), height=2)
        self.output_display.pack(fill="x", padx=5, pady=(5, 3))
        self.output_display.config(state="disabled")
        
        # Current Input display - NO LABEL, bigger font
        self.input_display = tk.Text(self, wrap="word", font=("Segoe UI", 16), height=1)
        self.input_display.pack(fill="x", padx=5, pady=3)
        self.input_display.config(state="disabled")
        
        # Next word predictions - NO LABEL, BIGGER buttons
        self.predictive_container = ttk.Frame(self)
        self.predictive_container.pack(fill="x", padx=5, pady=3)
        
        # Virtual keyboard - MAXIMIZED
        keyboard_frame = tk.Frame(self, bg=self.themes[self.current_theme]["bg"])
        keyboard_frame.pack(fill="both", expand=True, padx=5, pady=(3, 5))
        
        self.create_keyboard(keyboard_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Type → SPACE to add word → ENTER to speak & clear input | ◄=Backspace ►=Clear", relief="sunken", anchor="w", font=("Segoe UI", 8))
        self.status_bar.pack(fill="x", side="bottom")
        
        # Apply initial theme
        self.apply_theme()
        
        self.update_display()
    
    def update_display(self):
        """Update displays - fetch suggestions FIRST, then draw"""
        theme = self.themes[self.current_theme]
        
        # ============================================================
        # STEP 1: FETCH SUGGESTIONS FIRST before drawing anything
        # ============================================================
        if self.current_input:
            context_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
            context = get_context_words(" ".join(context_words), n=2)
            
            suggestions = ngram_model.get_completion_suggestions(
                self.current_input, context, max_results=5
            )
            
            print(f"🔍 Input: '{self.current_input}' → Suggestions: {suggestions}")
            
            if suggestions:
                self.current_completion = suggestions[0]
                self.alternative_suggestions = suggestions[1:5]
                print(f"   ✓ Completion: '{self.current_completion}'")
            else:
                self.current_completion = self.current_input
                self.alternative_suggestions = []
        else:
            self.current_completion = ""
            self.alternative_suggestions = []
        
        # ============================================================
        # STEP 2: DRAW OUTPUT DISPLAY (top textbox) with correct completion
        # ============================================================
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        
        # Show finalized words
        for i, word in enumerate(self.output_words):
            if i == self.output_cursor and self.current_input:
                continue
            else:
                self.output_display.insert("end", word + " ", "normal")
        
        if self.current_input:
            print(f"📊 Display - Input: '{self.current_input}', Completion: '{self.current_completion}'")
            
            # Show FULL completion in gray (could be multi-word like "thank you")
            if self.current_completion and self.current_completion != self.current_input:
                self.output_display.insert("end", self.current_completion, "suggestion")
                print(f"   ✓ Showing gray: '{self.current_completion}'")
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
        self.output_display.config(state="disabled")
        
        # ============================================================
        # STEP 3: DRAW INPUT DISPLAY (bottom textbox)
        # ============================================================
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
        
        # ============================================================
        # STEP 4: SHOW POPUP and PREDICTIONS
        # ============================================================
        if self.current_input and self.alternative_suggestions:
            self.show_alternative_popup()
        else:
            self.close_popup()
        
        self.update_predictions()
    
    def move_word_left(self):
        """Move cursor left in output (select previous word)"""
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        
        if self.output_cursor == -1:
            # Currently at end, move to last word
            self.output_cursor = len(self.output_words) - 1
        elif self.output_cursor > 0:
            self.output_cursor -= 1
        
        self.update_display()
        self.status_bar.config(text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'")
    
    def move_word_right(self):
        """Move cursor right in output (select next word)"""
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        
        if self.output_cursor == -1:
            self.status_bar.config(text="Already at end")
            return
        
        if self.output_cursor < len(self.output_words) - 1:
            self.output_cursor += 1
        else:
            # Move to end
            self.output_cursor = -1
        
        self.update_display()
        
        if self.output_cursor == -1:
            self.status_bar.config(text="Cursor at end - ready for new word")
        else:
            self.status_bar.config(text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'")
    
    def clear_selected_word(self):
        """Clear the word at cursor position"""
        if self.output_cursor == -1:
            self.status_bar.config(text="No word selected - cursor at end")
            return
        
        if self.output_cursor >= len(self.output_words):
            self.status_bar.config(text="No word at cursor")
            return
        
        # Remove word at cursor
        removed_word = self.output_words.pop(self.output_cursor)
        
        # Stay at same position
        # If we removed last word, move cursor to end
        if self.output_cursor >= len(self.output_words):
            self.output_cursor = -1
        
        self.update_display()
        self.status_bar.config(text=f"Removed '{removed_word}'")
    
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
        
        # Update prediction buttons
        if hasattr(self, 'predictive_container'):
            for widget in self.predictive_container.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.config(
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
        """Update next word prediction buttons - BIGGER"""
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        # Use output words for context
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
    
    def move_cursor_left(self):
        """Edit input - move cursor left in input (backspace)"""
        if self.current_input:
            # Remove last character (like backspace)
            self.current_input = self.current_input[:-1]
            self.update_display()
            self.status_bar.config(text="Removed last character from input")
        else:
            self.status_bar.config(text="Input is empty")
    
    def move_cursor_right(self):
        """Clear input without speaking"""
        if self.current_input:
            self.current_input = ""
            self.current_completion = ""
            self.alternative_suggestions = []
            self.update_display()
            self.status_bar.config(text="Input cleared (no speech)")
        else:
            self.status_bar.config(text="Input already empty")
    
    def update_cursor_display(self):
        """Update output display to show cursor position (deprecated - now in update_display)"""
        self.update_display()
    
    def finalize_word(self):
        """SPACE - finalize current word (replace at cursor or add at end)"""
        if not self.current_input:
            self.status_bar.config(text="Nothing to finalize")
            return
        
        # Close popup
        self.close_popup()
        
        # Use autocomplete or typed word
        word_to_add = self.current_completion if self.current_completion else self.current_input
        
        # DEBUG
        print(f"\n🔹 FINALIZE_WORD:")
        print(f"   current_input: '{self.current_input}'")
        print(f"   current_completion: '{self.current_completion}'")
        print(f"   word_to_add: '{word_to_add}'")
        
        # Get context
        context_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
        context = get_context_words(" ".join(context_words), n=2)
        
        # Learn and track
        if self.current_input != word_to_add:
            ngram_model.learn_from_user_typing(self.current_input, word_to_add)
        ngram_model.track_word_usage(word_to_add, context)
        
        # Replace or insert word
        if self.output_cursor == -1 or self.output_cursor >= len(self.output_words):
            # Add at end
            self.output_words.append(word_to_add)
            self.output_cursor = -1  # Stay at end
        else:
            # Replace word at cursor
            self.output_words[self.output_cursor] = word_to_add
            # Move cursor forward
            self.output_cursor += 1
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
        
        # Clear input
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.update_display()
        self.status_bar.config(text=f"Added '{word_to_add}'")
    
    def apply_alternative(self, word):
        """Apply alternative suggestion (when autocomplete gives error)"""
        # Close popup
        self.close_popup()
        
        output_text = " ".join(self.output_words)
        context = get_context_words(output_text, n=2)
        
        # Learn and track
        if self.current_input != word:
            ngram_model.learn_from_user_typing(self.current_input, word)
        ngram_model.track_word_usage(word, context)
        
        # Add word at cursor position
        if self.output_cursor == -1:
            self.output_words.append(word)
        else:
            self.output_words.insert(self.output_cursor, word)
            self.output_cursor += 1
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
        
        # Clear input
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.update_display()
        self.status_bar.config(text=f"Alternative selected: '{word}'")
    
    def apply_prediction(self, word):
        """Apply next word prediction"""
        output_text = " ".join(self.output_words)
        context = get_context_words(output_text, n=2)
        
        # Add predicted word at end
        self.output_words.append(word)
        self.output_cursor = -1  # Move cursor to end
        
        ngram_model.track_word_usage(word, context)
        
        self.update_display()
        self.status_bar.config(text=f"Predicted: '{word}'")
    
    def insert_char(self, char):
        """Insert character - just add to current input (word stays in list until space)"""
        self.current_input += char
        self.update_display()
        self.status_bar.config(text=f"Typing: '{self.current_input}'")
    
    def backspace(self):
        """Backspace - delete character from autocompleted word, current input, or pull back last word"""
        
        # DEBUG
        print(f"\n⌫ BACKSPACE:")
        print(f"   current_input: '{self.current_input}'")
        print(f"   current_completion: '{self.current_completion}'")
        print(f"   output_words: {self.output_words}")
        print(f"   output_cursor: {self.output_cursor}")
        
        if self.current_input:
            # MODE 1: Currently typing - ALWAYS just delete one character
            # Simple and foolproof - just remove last character from current_input
            self.current_input = self.current_input[:-1]
            self.current_completion = ""  # Clear to prevent loops
            self.alternative_suggestions = []
            print(f"   → Backspace: '{self.current_input}'")
            self.update_display()
            self.status_bar.config(text=f"Backspace")
        
        elif self.output_cursor == -1 and self.output_words:
            # MODE 2: Not typing, at end - pull back last word for editing
            last_word = self.output_words.pop()
            if len(last_word) > 1:
                self.current_input = last_word[:-1]
            else:
                self.current_input = ""
            print(f"   → Pulled back '{last_word}', now editing: '{self.current_input}'")
            self.update_display()
            self.status_bar.config(text=f"Editing: '{last_word}' → '{self.current_input}'")
        
        elif self.output_cursor != -1 and self.output_cursor < len(self.output_words):
            # MODE 3: Cursor on a word - delete that word
            deleted_word = self.output_words.pop(self.output_cursor)
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
            print(f"   → Deleted word: '{deleted_word}'")
            self.update_display()
            self.status_bar.config(text=f"Deleted word: '{deleted_word}'")
        
        else:
            # MODE 4: Nothing to delete
            print(f"   → Nothing to delete")
            self.status_bar.config(text="Nothing to delete")
    def enter(self):
        """ENTER - finalize any pending word, speak, and clear all"""
        # Finalize if typing
        if self.current_input:
            self.finalize_word()
        
        # Speak output
        output_text = " ".join(self.output_words)
        if output_text.strip():
            print(f"🔊 TTS: {output_text.strip()}")
            # TODO: Implement TTS with pyttsx3 or gTTS
        
        # Clear everything
        self.output_words = []
        self.output_cursor = -1
        self.current_input = ""
        
        self.update_display()
        self.status_bar.config(text="Spoken and cleared")
    
    def clear_all(self):
        """Clear everything"""
        self.output_words = []
        self.output_cursor = -1
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        
        self.update_display()
        self.status_bar.config(text="All cleared")
    
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
        
        # Row 0: LEFT ARROW | CLEAR ALL | SETTINGS | RIGHT ARROW (4 buttons)
        row0_frame = tk.Frame(main_container, bg=theme["bg"])
        row0_frame.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        row0_frame.grid_columnconfigure(0, weight=1, uniform="func")
        row0_frame.grid_columnconfigure(1, weight=3, uniform="func")
        row0_frame.grid_columnconfigure(2, weight=3, uniform="func")
        row0_frame.grid_columnconfigure(3, weight=1, uniform="func")
        row0_frame.grid_rowconfigure(0, weight=1)
        
        # Left arrow
        left_arrow_btn = tk.Button(
            row0_frame, text="◄",
            font=("Segoe UI", 20, "bold"),
            command=self.move_word_left,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        left_arrow_btn.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(left_arrow_btn)
        
        clear_all_btn = tk.Button(
            row0_frame, text="CLEAR ALL",
            font=("Segoe UI", 16, "bold"),
            command=self.clear_all,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        clear_all_btn.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(clear_all_btn)
        
        settings_btn = tk.Button(
            row0_frame, text="⚙ SETTINGS",
            font=("Segoe UI", 16, "bold"),
            command=self.show_settings,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        settings_btn.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(settings_btn)
        
        # Right arrow
        right_arrow_btn = tk.Button(
            row0_frame, text="►",
            font=("Segoe UI", 20, "bold"),
            command=self.move_word_right,
            relief="raised",
            bd=1,
            cursor="hand2"
        )
        right_arrow_btn.grid(row=0, column=3, sticky="nsew", padx=0, pady=0)
        self.keyboard_buttons.append(right_arrow_btn)
        
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