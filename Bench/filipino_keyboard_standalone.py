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
        "oo", "hindi", "salamat", "sorry"
    ]
    shortcuts = {"lng": "lang", "nmn": "naman"}
    corpus = ["kumusta ka", "mabuti naman"]
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
        
        # Check shortcuts
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if all_expansions:
            for full_word, source in all_expansions:
                priority_multiplier = 3.0 if source == 'user' else 2.0
                shortcut_candidates.append((full_word, True, True, priority_multiplier))
        
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
        
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        
        for word in exact_matches:
            if word in char_completions:
                candidates.append((word, True, False, 1.5))
            else:
                candidates.append((word, True, False, 1.0))
        
        for word in char_completions:
            if word not in exact_matches and len(word) >= min_word_length and word.isalpha() and self._has_vowels(word):
                candidates.append((word, True, False, 1.3))
        
        # Add typo tolerance - find near matches using Damerau-Levenshtein
        if len(exact_matches) < max_results:
            for word in self.vocabulary:
                if word not in exact_matches:
                    # Skip very short words
                    if len(word) < min_word_length:
                        continue
                    
                    # Skip words with punctuation
                    if not word.isalpha():
                        continue
                    
                    # Skip acronyms without vowels
                    if not self._has_vowels(word):
                        continue
                    
                    # Only consider words of similar length (within 3 chars)
                    if abs(len(word) - len(prefix)) <= 3:
                        distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                        # Only include if distance is small (1-2 edits)
                        if distance <= 2:
                            candidates.append((word, False, False, 1.0))  # Not exact, not shortcut
        
        # Combine
        all_candidates = shortcut_candidates + candidates
        
        # Score with DL distance component
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
            prob = self.get_word_probability(word, context)
            
            # Damerau-Levenshtein similarity score
            dl_distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
            dl_score = 1.0 / (1.0 + dl_distance)  # Convert distance to similarity
            
            prefix_ratio = len(prefix) / len(word)
            exact_bonus = 2.0 if is_exact_match else 1.0
            shortcut_bonus = 5.0 * priority_mult if is_shortcut else 1.0
            
            usage_bonus = 1.0
            if is_shortcut and prefix in self.user_shortcut_usage:
                usage_count = self.user_shortcut_usage[prefix]
                usage_bonus = 1.0 + min(usage_count / 10.0, 2.0)
            
            # Updated scoring with DL component
            final_score = (
                prob * 0.70 +          # 70% n-gram probability
                dl_score * 0.30 +      # 20% edit distance 
                prefix_ratio * 0.10    # 10% prefix coverage
            ) * exact_bonus * shortcut_bonus * usage_bonus
            
            scored.append((word, final_score))
        
        scored.sort(key=lambda x: -x[1])
        return [word for word, score in scored[:max_results]]
    
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

class FilipinoKeyboard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Filipino Keyboard - Standalone")
        self.geometry("1200x850")
        self.configure(bg="#f0f0f0")
        
        self.create_widgets()
        
        # Bind space key in input_display to commit autocomplete
        self.input_display.bind("<space>", self.on_space_pressed)

    def create_widgets(self):
        # --------------------------
        # Output display (autocomplete)
        # --------------------------
        output_frame = ttk.Frame(self)
        output_frame.pack(fill="both", expand=False, padx=15, pady=(15,5))
        
        ttk.Label(output_frame, text="Output (Autocompleted):", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.output_display = tk.Text(output_frame, wrap="word", font=("Segoe UI", 16), height=6, state="disabled", bg="#e0e0e0")
        self.output_display.pack(fill="both", expand=True)
        
        # --------------------------
        # Input text box
        # --------------------------
        input_frame = ttk.Frame(self)
        input_frame.pack(fill="both", expand=False, padx=15, pady=(5,15))
        
        ttk.Label(input_frame, text="Input:", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.input_display = tk.Text(input_frame, wrap="word", font=("Segoe UI", 16), height=6)
        self.input_display.pack(fill="both", expand=True)
        self.input_display.bind('<KeyRelease>', lambda e: self.update_autocomplete())
        
        # --------------------------
        # Virtual keyboard
        # --------------------------
        keyboard_frame = ttk.LabelFrame(self, text="Virtual Keyboard", padding="10")
        keyboard_frame.pack(fill="both", expand=False, padx=15, pady=10)
        self.create_keyboard(keyboard_frame)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief="sunken", anchor="w", font=("Segoe UI", 9))
        self.status_bar.pack(fill="x", side="bottom")
    
    # --------------------------
    # Autocomplete logic
    # --------------------------
    def update_autocomplete(self):
        """Show suggestion in output display without touching input box"""
        text = self.input_display.get("1.0", "end-1c")
        current_token = get_current_token(text)
        context = get_context_words(text, n=2)

        if current_token:
            completions = ngram_model.get_completion_suggestions(current_token, context, max_results=1)
            suggestion = completions[0] if completions else current_token
        else:
            suggestion = ""

        # Build display text: previous words + suggested token
        words = text.strip().split()
        if words:
            if text.endswith(" "):
                # User already pressed space, no suggestion needed
                display_text = text.strip()
            else:
                display_text = " ".join(words[:-1] + [suggestion])
        else:
            display_text = suggestion

        # Update output box
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        self.output_display.insert("1.0", display_text)
        self.output_display.config(state="disabled")

        self.status_bar.config(text=f"Suggestion: '{suggestion}'")


    def on_space_pressed(self):
        """Commit the current suggestion and insert a space"""
        text = self.input_display.get("1.0", "end-1c")
        current_token = get_current_token(text)
        context = get_context_words(text, n=2)

        if current_token:
            # Get best suggestion for current token
            completions = ngram_model.get_completion_suggestions(current_token, context, max_results=1)
            commit_word = completions[0] if completions else current_token

            # Replace the current token with the committed word
            words = text.strip().split()
            new_text = " ".join(words[:-1] + [commit_word]) + " "
        else:
            new_text = text + " "

        # Update input display only when committing
        self.input_display.delete("1.0", "end")
        self.input_display.insert("1.0", new_text)

        # Refresh autocomplete display for next word
        self.update_autocomplete()

     # --------------------------
    # Virtual keyboard functions
    # --------------------------
    def insert_char(self, char):
        self.input_display.insert("end", char)
        self.update_autocomplete()
    
    def backspace(self):
        self.input_display.delete("end-2c", "end-1c")
        self.update_autocomplete()
    
    def space(self):
        """Virtual keyboard space button"""
        self.on_space_pressed()
    
    def enter(self):
        self.input_display.insert("end", "\n")
        self.update_autocomplete()
    
    def create_keyboard(self, parent):
        style = ttk.Style()
        style.configure("Keyboard.TButton", font=("Segoe UI", 11), padding=8)
        
        # Number row
        row_num = ttk.Frame(parent)
        row_num.pack(fill="x", pady=3)
        for ch in "1234567890":
            btn = ttk.Button(row_num, text=ch, style="Keyboard.TButton",
                             command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", expand=True, fill="x", padx=2)
        
        # Q-P row + Backspace
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=3)
        for ch in "qwertyuiop":
            btn = ttk.Button(row1, text=ch.upper(), style="Keyboard.TButton",
                             command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", expand=True, fill="x", padx=2)
        backspace_btn = ttk.Button(row1, text="⌫", style="Keyboard.TButton", command=self.backspace)
        backspace_btn.pack(side="left", expand=True, fill="x", padx=2)
        
        # A-L row + Enter
        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=3)
        for ch in "asdfghjkl":
            btn = ttk.Button(row2, text=ch.upper(), style="Keyboard.TButton",
                             command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", expand=True, fill="x", padx=2)
        enter_btn = ttk.Button(row2, text="↵", style="Keyboard.TButton", command=self.enter)
        enter_btn.pack(side="left", expand=True, fill="x", padx=2)
        
        # Z-M row
        row3 = ttk.Frame(parent)
        row3.pack(fill="x", pady=3)
        for ch in "zxcvbnm":
            btn = ttk.Button(row3, text=ch.upper(), style="Keyboard.TButton",
                             command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", expand=True, fill="x", padx=2)
        
        # Space row
        bottom_row = ttk.Frame(parent)
        bottom_row.pack(fill="x", pady=3)
        space_btn = ttk.Button(bottom_row, text="SPACE", style="Keyboard.TButton", command=self.space)
        space_btn.pack(expand=True, fill="x", padx=3, ipadx=80, ipady=12)

if __name__ == "__main__":
    app = FilipinoKeyboard()
    app.mainloop()
