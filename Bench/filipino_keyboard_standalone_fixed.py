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
DATASET_FILE = "filipino_dataset.json"  # NEW: External dataset file
MODEL_VERSION = "2.1_json_dataset"  # Updated version

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
        
        # Flatten vocabulary
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

# Load dataset at startup
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
    
    # Create distance matrix
    d = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1
    
    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            
            d[(i, j)] = min(
                d[(i-1, j)] + 1,      # deletion
                d[(i, j-1)] + 1,      # insertion
                d[(i-1, j-1)] + cost, # substitution
            )
            
            # Transposition
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
        
        # Character-level n-grams
        self.char_bigrams = defaultdict(Counter)
        self.char_trigrams = defaultdict(Counter)
        
        # Shortcuts
        self.csv_shortcuts = {}
        self.user_shortcuts = {}
        self.user_shortcut_usage = Counter()
        self.new_words = set()
        self.word_usage_history = []
    
    def train_from_builtin(self):
        """Train from built-in Filipino vocabulary (NO external datasets!)"""
        print("📚 Loading built-in Filipino vocabulary...")
        
        # Add all words multiple times to simulate corpus
        all_tokens = []
        
        # Add conversational words (high frequency)
        for word in FILIPINO_WORDS:
            # Add each word 10 times to give it good frequency
            all_tokens.extend([word.lower()] * 10)
        
        # Add communication corpus (IMPORTANT for predictions!)
        print("📝 Processing communication corpus...")
        for phrase in COMMUNICATION_CORPUS:
            # Split phrase into words
            words = phrase.lower().split()
            # Add phrase multiple times (50x) for VERY strong n-gram patterns
            for _ in range(50):
                all_tokens.extend(words)
        
        # Add shortcuts as vocabulary
        for shortcut, full_word in FILIPINO_SHORTCUTS.items():
            self.csv_shortcuts[shortcut] = full_word
            # Add full words too
            all_tokens.extend([full_word.lower()] * 5)
        
        print(f"✓ Built-in words: {len(FILIPINO_WORDS)}")
        print(f"✓ Communication phrases: {len(COMMUNICATION_CORPUS)}")
        print(f"✓ Built-in shortcuts: {len(FILIPINO_SHORTCUTS)}")
        print(f"✓ Total tokens for training: {len(all_tokens)}")
        
        # Build vocabulary
        self.vocabulary.update(all_tokens)
        
        # Build n-grams
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
            'version': MODEL_VERSION,  # Track version
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
            
            # Check version - rebuild if outdated
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
        """Get word completion suggestions"""
        prefix = prefix.lower()
        
        # DEBUG: Print what we're looking for
        if prefix == "nlng":
            print(f"\n🔍 DEBUG: Looking for '{prefix}'")
            print(f"  csv_shortcuts has nlng? {'nlng' in self.csv_shortcuts}")
            if 'nlng' in self.csv_shortcuts:
                print(f"  nlng maps to: {self.csv_shortcuts['nlng']}")
            print(f"  Total shortcuts: {len(self.csv_shortcuts)}")
        
        # PRIORITY 1: Check shortcuts (HIGHEST PRIORITY)
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if prefix == "nlng":
            print(f"  Expansions found: {all_expansions}")
        
        if all_expansions:
            for full_word, source in all_expansions:
                priority_multiplier = 10.0 if source == 'user' else 8.0
                shortcut_candidates.append((full_word, False, True, priority_multiplier))
        
        if prefix == "nlng":
            print(f"  Shortcut candidates: {shortcut_candidates}")
        
        # PRIORITY 2: Exact prefix matches
        candidates = []
        
        # Minimum word length
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
        
        if prefix == "nlng":
            print(f"  Exact matches: {exact_matches}")
        
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        
        # Add exact prefix matches
        for word in exact_matches:
            if word in char_completions:
                candidates.append((word, True, False, 2.0))
            else:
                candidates.append((word, True, False, 1.0))
        
        # PRIORITY 3: Fuzzy matches (ONLY if no exact matches or shortcuts)
        fuzzy_candidates = []
        if not shortcut_candidates and len(exact_matches) < 3 and len(prefix) >= 2:
            for word in self.vocabulary:
                if word in exact_matches:
                    continue
                
                # Skip very short words
                if len(word) < min_word_length:
                    continue
                
                # Skip words with punctuation
                if not word.isalpha():
                    continue
                
                # Skip acronyms without vowels
                if not self._has_vowels(word):
                    continue
                
                # Check if word contains the prefix
                if prefix in word:
                    position = word.index(prefix)
                    fuzzy_candidates.append((word, False, False, 0.3 / (position + 1)))
                    continue
                
                # Only consider words of similar length
                if abs(len(word) - len(prefix)) <= 2:
                    distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                    # Only if very close (1 edit)
                    if distance <= 1:
                        fuzzy_candidates.append((word, False, False, 0.1))
        
        # Combine all candidates
        all_candidates = shortcut_candidates + candidates + fuzzy_candidates
        
        if prefix == "nlng":
            print(f"  Total candidates: {len(all_candidates)}")
            print(f"  All candidates: {[w for w, _, _, _ in all_candidates]}")
        
        # Score all candidates
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
            prob = self.get_word_probability(word, context)
            
            # Calculate scores
            if is_shortcut:
                # Shortcuts get MASSIVE boost
                final_score = prob * priority_mult * 100
            elif is_exact_match:
                # Exact prefix matches get good boost
                final_score = prob * priority_mult * 10
            else:
                # Fuzzy matches get small boost
                final_score = prob * priority_mult
            
            # Extra boost for frequently used shortcuts
            if is_shortcut and prefix in self.user_shortcut_usage:
                usage_count = self.user_shortcut_usage[prefix]
                final_score *= (1.0 + min(usage_count / 10.0, 2.0))
            
            scored.append((word, final_score))
        
        # Remove duplicates (keep highest score)
        seen = {}
        for word, score in scored:
            if word not in seen or score > seen[word]:
                seen[word] = score
        
        # Sort by score
        unique_scored = [(word, score) for word, score in seen.items()]
        unique_scored.sort(key=lambda x: -x[1])
        
        result = [word for word, score in unique_scored[:max_results]]
        
        if prefix == "nlng":
            print(f"  Final result: {result}\n")
        
        return result
    
    def get_next_word_suggestions(self, context=None, max_results=6):
        """Predict next word"""
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
print("FILIPINO KEYBOARD - STANDALONE VERSION")
print("Built-in vocabulary only (no external datasets!)")
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
# GUI
# =============================================================================
class FilipinoKeyboard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Filipino Keyboard - Standalone")
        self.geometry("1200x850")  # Increased from 900x700
        self.configure(bg="#f0f0f0")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Text display - smaller height
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=False, padx=15, pady=15)  # expand=False
        
        self.text_display = tk.Text(text_frame, wrap="word", font=("Segoe UI", 16), height=6)  # Reduced from 12 to 6
        self.text_display.pack(fill="both", expand=False)  # expand=False
        self.text_display.bind('<KeyRelease>', lambda e: self.update_suggestions())
        
        # Suggestions - more space
        suggestions_frame = ttk.LabelFrame(self, text="Suggestions", padding="15")
        suggestions_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(suggestions_frame, text="Word Completion:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.completion_container = ttk.Frame(suggestions_frame)
        self.completion_container.pack(fill="x", pady=8)
        
        ttk.Label(suggestions_frame, text="Next Word:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(15,0))
        self.predictive_container = ttk.Frame(suggestions_frame)
        self.predictive_container.pack(fill="x", pady=8)
        
        # Virtual keyboard - more spacing
        keyboard_frame = ttk.LabelFrame(self, text="Virtual Keyboard", padding="15")
        keyboard_frame.pack(fill="both", padx=15, pady=10)
        
        self.create_keyboard(keyboard_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief="sunken", anchor="w", font=("Segoe UI", 9))
        self.status_bar.pack(fill="x", side="bottom")
        
        self.update_suggestions()
    
    def update_suggestions(self):
        """Update suggestions"""
        for widget in self.completion_container.winfo_children():
            widget.destroy()
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        # Word completion
        if token:
            completions = ngram_model.get_completion_suggestions(token, context, max_results=5)
            
            if completions:
                for word in completions[:8]:
                    display_word = word.capitalize() if token and token[0].isupper() else word
                    
                    btn = ttk.Button(
                        self.completion_container,
                        text=display_word,
                        command=lambda w=display_word: self.apply_completion(w),
                        style="Suggestion.TButton"
                    )
                    btn.pack(side="left", padx=5, pady=5, ipadx=12, ipady=8)
        
        # Next word prediction
        predictions = ngram_model.get_next_word_suggestions(context, max_results=5)
        
        if predictions:
            for word in predictions[:6]:
                btn = ttk.Button(
                    self.predictive_container,
                    text=word,
                    command=lambda w=word: self.apply_prediction(w),
                    style="Suggestion.TButton"
                )
                btn.pack(side="left", padx=5, pady=5, ipadx=12, ipady=8)
    
    def apply_completion(self, word):
        """Apply completion"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        if token:
            ngram_model.learn_from_user_typing(token, word)
            
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            
            self.text_display.delete(start_index, "end-1c")
            self.text_display.insert(start_index, word + " ")
        else:
            self.text_display.insert("end", word + " ")
        
        ngram_model.track_word_usage(word, context)
        
        self.update_suggestions()
        self.status_bar.config(text=f"Applied: '{word}'")
    
    def apply_prediction(self, word):
        """Apply prediction"""
        text = self.text_display.get("1.0", "end-1c")
        context = get_context_words(text, n=2)
        
        if text and not text.endswith(" "):
            self.text_display.insert("end", " ")
        
        self.text_display.insert("end", word + " ")
        
        ngram_model.track_word_usage(word, context)
        
        self.update_suggestions()
        self.status_bar.config(text=f"Predicted: '{word}'")
    
    def insert_char(self, char):
        """Insert character"""
        self.text_display.insert("end", char)
        self.update_suggestions()
    
    def backspace(self):
        """Backspace"""
        self.text_display.delete("end-2c", "end-1c")
        self.update_suggestions()
    
    def space(self):
        """Space"""
        self.text_display.insert("end", " ")
        self.update_suggestions()
    
    def enter(self):
        """Enter"""
        self.text_display.insert("end", "\n")
        self.update_suggestions()
    
    def create_keyboard(self, parent):
        """Create virtual keyboard"""
        style = ttk.Style()
        style.configure("Keyboard.TButton", font=("Segoe UI", 11), padding=8)
        style.configure("Suggestion.TButton", font=("Segoe UI", 11), padding=6)
        
        parent.grid_columnconfigure(0, weight=1)
        
        # Function row
        func_row = ttk.Frame(parent)
        func_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        
        clear_btn = ttk.Button(func_row, text="Clear", 
                             style="Keyboard.TButton",
                             command=lambda: [self.text_display.delete("1.0", "end"), 
                                            self.update_suggestions()])
        clear_btn.pack(side="left", padx=3, ipadx=8, ipady=4, expand=True, fill="x")
        
        # Number row
        row_num = ttk.Frame(parent)
        row_num.grid(row=1, column=0, sticky="ew", pady=3)
        
        for ch in "1234567890":
            btn = ttk.Button(row_num, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", ipadx=12, ipady=8, expand=True, fill="x", padx=2)
        
        # First letter row: Q-P + Backspace
        row1 = ttk.Frame(parent)
        row1.grid(row=2, column=0, sticky="ew", pady=3)
        
        for ch in "qwertyuiop":
            btn = ttk.Button(row1, text=ch.upper(), style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", ipadx=12, ipady=8, expand=True, fill="x", padx=2)
        
        # Backspace after P
        backspace_btn = ttk.Button(row1, text="⌫", style="Keyboard.TButton",
                                   command=self.backspace)
        backspace_btn.pack(side="left", ipadx=15, ipady=8, padx=2)
        
        # Second letter row: A-L + Enter
        row2 = ttk.Frame(parent)
        row2.grid(row=3, column=0, sticky="ew", pady=3)
        
        for ch in "asdfghjkl":
            btn = ttk.Button(row2, text=ch.upper(), style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", ipadx=12, ipady=8, expand=True, fill="x", padx=2)
        
        # Enter after L
        enter_btn = ttk.Button(row2, text="↵", style="Keyboard.TButton",
                              command=self.enter)
        enter_btn.pack(side="left", ipadx=15, ipady=8, padx=2)
        
        # Third letter row: Z-M
        row3 = ttk.Frame(parent)
        row3.grid(row=4, column=0, sticky="ew", pady=3)
        
        for ch in "zxcvbnm":
            btn = ttk.Button(row3, text=ch.upper(), style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", ipadx=12, ipady=8, expand=True, fill="x", padx=2)
        
        # Bottom row - just SPACE
        bottom_row = ttk.Frame(parent)
        bottom_row.grid(row=5, column=0, sticky="ew", pady=3)
        
        space_btn = ttk.Button(bottom_row, text="SPACE", style="Keyboard.TButton",
                              command=self.space)
        space_btn.pack(ipadx=80, ipady=12, expand=True, fill="x", padx=3)

if __name__ == "__main__":
    app = FilipinoKeyboard()
    app.mainloop()
