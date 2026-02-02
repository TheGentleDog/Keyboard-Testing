import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
from datasets import load_dataset
import pickle

# Configuration
MAX_SUGGESTIONS = 8
NGRAM_CACHE_FILE = "ngram_model.pkl"
VOCAB_CACHE_FILE = "vocabulary.pkl"

# ---------------------------------------------------------------------
# N-GRAM LANGUAGE MODEL
# ---------------------------------------------------------------------
class NgramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.vocabulary = set()
        self.total_words = 0
        self.learned_shortcuts = {}  # Maps shortcuts to full words
        self.shortcut_frequencies = Counter()  # Track how often shortcuts appear
        
    def train_from_dataset(self):
        """Load and train n-gram model from TLUnified NER dataset"""
        print("Loading TLUnified NER dataset...")
        
        try:
            # Load the validation split
            dataset = load_dataset("ljvmiranda921/tlunified-ner", split="validation")
            print(f"✓ Loaded {len(dataset)} examples")
            
            # Extract tokens and build n-grams
            for example in dataset:
                tokens = example['tokens']
                
                # Clean tokens (lowercase for model consistency)
                clean_tokens = [token.lower() for token in tokens if token.strip()]
                
                # Update vocabulary
                self.vocabulary.update(clean_tokens)
                
                # Build n-grams
                for i, token in enumerate(clean_tokens):
                    # Unigram
                    self.unigrams[token] += 1
                    self.total_words += 1
                    
                    # Bigram
                    if i > 0:
                        prev = clean_tokens[i-1]
                        self.bigrams[prev][token] += 1
                    
                    # Trigram
                    if i > 1:
                        prev2 = clean_tokens[i-2]
                        prev1 = clean_tokens[i-1]
                        context = (prev2, prev1)
                        self.trigrams[context][token] += 1
            
            print(f"✓ Built vocabulary: {len(self.vocabulary)} unique words")
            print(f"✓ Total words processed: {self.total_words}")
            print(f"✓ Bigram contexts: {len(self.bigrams)}")
            print(f"✓ Trigram contexts: {len(self.trigrams)}")
            
            # Learn shortcuts from the dataset
            print("\n🔍 Learning shortcuts from dataset...")
            self._learn_shortcuts_from_vocabulary()
            
        except Exception as e:
            print(f"⚠ Error loading dataset: {e}")
            print("Using fallback vocabulary...")
            self._load_fallback()
    
    def _load_fallback(self):
        """Minimal fallback if dataset fails"""
        fallback = [
            "ang", "ng", "sa", "mga", "na", "ay", "at", "si", "ni", "kay",
            "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
            "ito", "iyan", "iyon", "dito", "doon", "diyan",
            "may", "wala", "hindi", "oo", "bakit", "paano", "kailan", "saan",
            "kung", "dahil", "pero", "o", "para", "tungkol",
            "malaki", "maliit", "maganda", "pangit", "masaya", "malungkot",
            "kumain", "uminom", "matulog", "maglaro", "magbasa", "magsulat",
            "bahay", "eskwela", "trabaho", "pamilya", "kaibigan", "pagkain"
        ]
        self.vocabulary = set(fallback)
        for word in fallback:
            self.unigrams[word] = 10
        self.total_words = len(fallback) * 10
    
    def _learn_shortcuts_from_vocabulary(self):
        """
        Learn common shortcut patterns from the vocabulary.
        Also adds bigram support for shortcuts.
        """
        def remove_vowels(word):
            """Remove vowels but keep at least 2 characters"""
            vowels = set('aeiouAEIOU')
            consonants = ''.join([c for c in word if c not in vowels])
            return consonants if len(consonants) >= 2 else None
        
        def create_number_shortcut(word):
            
            replacements = {'to': '2', 'too': '2', 'for': '4', 'ate': '8'}
            for pattern, num in replacements.items():
                if pattern in word:
                    return word.replace(pattern, num)
            return None
        
        # Analyze vocabulary
        shortcut_candidates = defaultdict(list)
        
        print(f"  Analyzing {len(self.vocabulary)} words...")
        
        for word in self.vocabulary:
            if len(word) < 3:
                continue
            
            freq = self.unigrams.get(word, 0)
            # LOWER THRESHOLD: Learn from words with freq >= 3
            if freq < 3:
                continue
            
            shortcuts = []
            
            # Pattern 1: Remove vowels
            vowel_removed = remove_vowels(word)
            if vowel_removed and len(vowel_removed) < len(word) and len(vowel_removed) >= 2:
                shortcuts.append(vowel_removed)
            
            # Pattern 2: Number substitution  
            num_shortcut = create_number_shortcut(word)
            if num_shortcut and num_shortcut != word:
                shortcuts.append(num_shortcut)
            
            # Pattern 3: First chars (for longer words)
            if len(word) >= 4:
                shortcuts.append(word[:3])
                shortcuts.append(word[:2])
            
            # Add shortcuts (must be at least 1 char shorter)
            for shortcut in shortcuts:
                if len(shortcut) < len(word):
                    shortcut_candidates[shortcut.lower()].append((word, freq))
        
        print(f"  Found {len(shortcut_candidates)} potential patterns")
        
        # Filter shortcuts
        for shortcut, word_list in shortcut_candidates.items():
            if len(shortcut) < 2:
                continue
            
            word_list.sort(key=lambda x: x[1], reverse=True)
            
            # Accept if: (1) only one word, (2) dominant word, or (3) high frequency
            if len(word_list) == 1:
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
            elif len(word_list) > 1 and word_list[0][1] >= word_list[1][1] * 2:
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
            elif len(word_list) > 1 and word_list[0][1] > 20:  # High freq word
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
        
        print(f"✓ Learned {len(self.learned_shortcuts)} shortcuts from dataset")
        
        # FALLBACK: Add common Filipino shortcuts manually
        manual_shortcuts = {
            'lng': 'lang', 'nmn': 'naman', 'ksi': 'kasi', 'kng': 'kung',
            'd2': 'dito', 'dn': 'doon', 'dyn': 'diyan',
            'pde': 'pwede', 'pwd': 'pwede', 'sna': 'sana', 'tlg': 'talaga',
            'sya': 'siya', 'xa': 'siya', 'aq': 'ako', 'u': 'ikaw',
            'nde': 'hindi', 'hnd': 'hindi', 'pra': 'para'
        }
        
        # Add manual shortcuts that weren't learned
        added_manual = 0
        for shortcut, full_word in manual_shortcuts.items():
            if shortcut not in self.learned_shortcuts:
                self.learned_shortcuts[shortcut] = full_word
                self.shortcut_frequencies[shortcut] = 50
                added_manual += 1
        
        if added_manual > 0:
            print(f"✓ Added {added_manual} common Filipino shortcuts manually")
        
        # Show learned shortcuts
        if self.learned_shortcuts:
            print(f"\n📝 Total shortcuts: {len(self.learned_shortcuts)}")
            sorted_shortcuts = sorted(
                self.learned_shortcuts.items(),
                key=lambda x: self.shortcut_frequencies[x[0]],
                reverse=True
            )
            print("   Top 15:")
            for shortcut, full_word in sorted_shortcuts[:15]:
                freq = self.shortcut_frequencies[shortcut]
                print(f"   '{shortcut}' → '{full_word}' (freq: {freq})")
    
    def resolve_shortcut(self, word):
        """
        Resolve a word to its full form if it's a shortcut.
        Used for bigram/trigram context lookup.
        
        Example: resolve_shortcut('lng') → 'lang'
        """
        word_lower = word.lower()
        if word_lower in self.learned_shortcuts:
            return self.learned_shortcuts[word_lower]
        return word
    
    def save_cache(self):
        """Save trained model to disk"""
        data = {
            'unigrams': dict(self.unigrams),
            'bigrams': {k: dict(v) for k, v in self.bigrams.items()},
            'trigrams': {k: dict(v) for k, v in self.trigrams.items()},
            'vocabulary': list(self.vocabulary),
            'total_words': self.total_words,
            'learned_shortcuts': self.learned_shortcuts,
            'shortcut_frequencies': dict(self.shortcut_frequencies)
        }
        with open(NGRAM_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved n-gram model to {NGRAM_CACHE_FILE}")
        print(f"✓ Saved {len(self.learned_shortcuts)} learned shortcuts")
    
    def load_cache(self):
        """Load trained model from disk"""
        if not os.path.exists(NGRAM_CACHE_FILE):
            return False
        
        try:
            with open(NGRAM_CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            
            self.unigrams = Counter(data['unigrams'])
            self.bigrams = defaultdict(Counter)
            for k, v in data['bigrams'].items():
                self.bigrams[k] = Counter(v)
            
            self.trigrams = defaultdict(Counter)
            for k, v in data['trigrams'].items():
                self.trigrams[k] = Counter(v)
            
            self.vocabulary = set(data['vocabulary'])
            self.total_words = data['total_words']
            
            # Load learned shortcuts if available
            self.learned_shortcuts = data.get('learned_shortcuts', {})
            self.shortcut_frequencies = Counter(data.get('shortcut_frequencies', {}))
            
            print(f"✓ Loaded cached n-gram model")
            print(f"  Vocabulary: {len(self.vocabulary)} words")
            print(f"  Total words: {self.total_words}")
            print(f"  Learned shortcuts: {len(self.learned_shortcuts)}")
            return True
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            return False
    
    def get_word_probability(self, word, context=None):
        """Calculate probability of word given context using n-gram model"""
        word = word.lower()
        
        # Laplace smoothing parameter
        alpha = 0.1
        
        if context is None or len(context) == 0:
            # Unigram probability with smoothing
            count = self.unigrams.get(word, 0)
            vocab_size = len(self.vocabulary)
            return (count + alpha) / (self.total_words + alpha * vocab_size)
        
        elif len(context) == 1:
            # Bigram probability with smoothing
            prev = context[0].lower()
            
            # SHORTCUT SUPPORT: Resolve previous word if it's a shortcut
            prev = self.resolve_shortcut(prev)
            
            count = self.bigrams[prev].get(word, 0)
            prev_count = self.unigrams.get(prev, 0)
            vocab_size = len(self.vocabulary)
            
            if prev_count == 0:
                # Backoff to unigram
                return self.get_word_probability(word)
            
            return (count + alpha) / (prev_count + alpha * vocab_size)
        
        else:  # len(context) >= 2
            # Trigram probability with smoothing
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            
            # SHORTCUT SUPPORT: Resolve both context words if they're shortcuts
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            
            trigram_context = (prev2, prev1)
            
            count = self.trigrams[trigram_context].get(word, 0)
            context_count = sum(self.trigrams[trigram_context].values())
            vocab_size = len(self.vocabulary)
            
            if context_count == 0:
                # Backoff to bigram
                return self.get_word_probability(word, [prev1])
            
            return (count + alpha) / (context_count + alpha * vocab_size)
    
    def get_completion_suggestions(self, prefix, context=None, max_results=8):
        """
        Get word completions using hierarchical ranking:
        
        PRIORITY ORDER:
        1. N-gram probability (70%) - Statistical language model based on context
        2. Damerau-Levenshtein distance (20%) - Typo tolerance via edit distance
        3. Prefix match quality (10%) - How much of the word is typed
        
        Process:
        - First: Find exact prefix matches (prioritized)
        - Second: Find near-matches via edit distance (typo correction)
        - Then: Rank all candidates by weighted score with N-gram as primary factor
        """
        prefix = prefix.lower()
        
        # PRIORITY 0: Check learned shortcuts FIRST
        shortcut_candidates = []
        if prefix in self.learned_shortcuts:
            # Direct shortcut match - this gets highest priority
            full_word = self.learned_shortcuts[prefix]
            shortcut_candidates.append((full_word, True, True))  # word, exact_match, is_shortcut
        
        # Find all words that start with prefix OR are close via edit distance
        candidates = []
        
        # Strategy 1: Exact prefix matches (best candidates)
        exact_matches = [word for word in self.vocabulary if word.startswith(prefix)]
        candidates.extend([(word, True, False) for word in exact_matches])  # Not shortcuts
        
        # Strategy 2: Typo tolerance - find words with small edit distance
        # Only if we have few exact matches
        if len(exact_matches) < max_results:
            for word in self.vocabulary:
                if word not in exact_matches:
                    # Only consider words of similar length (within 2 chars)
                    if abs(len(word) - len(prefix)) <= 2:
                        distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                        # Only include if distance is small (1-2 edits)
                        if distance <= 2:
                            candidates.append((word, False, False))  # Not exact, not shortcut
        
        # Combine shortcut candidates with regular candidates
        all_candidates = shortcut_candidates + candidates
        
        # Score each candidate
        scored = []
        for word, is_exact_match, is_shortcut in all_candidates:
            # 1. N-gram probability
            prob = self.get_word_probability(word, context)
            
            # 2. Damerau-Levenshtein distance score
            dl_distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
            # Convert distance to similarity score (0 = perfect match)
            dl_score = 1.0 / (1.0 + dl_distance)
            
            # 3. Prefix match quality
            prefix_ratio = len(prefix) / len(word)
            
            # 4. Exact match bonus
            exact_bonus = 2.0 if is_exact_match else 1.0
            
            # 5. SHORTCUT BONUS - Highest priority!
            shortcut_bonus = 5.0 if is_shortcut else 1.0
            
            # Combined score with weighted factors
            # PRIMARY: N-gram probability (70%) - language model comes first
            # SECONDARY: Damerau-Levenshtein similarity (20%) - typo tolerance
            # TERTIARY: Prefix ratio (10%) - completion quality
            final_score = (
                prob * 0.7 +           # 70% n-gram probability (PRIMARY)
                dl_score * 0.2 +       # 20% edit distance similarity (SECONDARY)
                prefix_ratio * 0.1     # 10% prefix coverage (TERTIARY)
            ) * exact_bonus * shortcut_bonus  # Bonuses multiply
            
            scored.append((word, final_score, dl_distance, is_shortcut))
        
        # Sort by score descending, then by edit distance ascending
        scored.sort(key=lambda x: (-x[1], x[2]))
        
        # Return top results
        return [word for word, score, dist, is_shortcut in scored[:max_results]]
    
    def get_next_word_suggestions(self, context=None, max_results=6):
        """Predict next word based on context using n-gram probabilities"""
        if context is None or len(context) == 0:
            # Return most common words
            most_common = self.unigrams.most_common(max_results)
            return [word for word, count in most_common]
        
        elif len(context) == 1:
            # Use bigram model
            prev = context[0].lower()
            
            # SHORTCUT SUPPORT: Resolve shortcut before bigram lookup
            prev = self.resolve_shortcut(prev)
            
            if prev in self.bigrams:
                most_common = self.bigrams[prev].most_common(max_results)
                return [word for word, count in most_common]
            else:
                # Backoff to unigram
                most_common = self.unigrams.most_common(max_results)
                return [word for word, count in most_common]
        
        else:  # len(context) >= 2
            # Use trigram model
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            
            # SHORTCUT SUPPORT: Resolve shortcuts before trigram lookup
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            
            trigram_context = (prev2, prev1)
            
            if trigram_context in self.trigrams:
                most_common = self.trigrams[trigram_context].most_common(max_results)
                return [word for word, count in most_common]
            else:
                # Backoff to bigram
                if prev1 in self.bigrams:
                    most_common = self.bigrams[prev1].most_common(max_results)
                    return [word for word, count in most_common]
                else:
                    # Backoff to unigram
                    most_common = self.unigrams.most_common(max_results)
                    return [word for word, count in most_common]

# ---------------------------------------------------------------------
# DAMERAU-LEVENSHTEIN DISTANCE
# ---------------------------------------------------------------------
def damerau_levenshtein_distance(s1, s2):
    """
    Calculate Damerau-Levenshtein distance between two strings.
    Includes transpositions (character swaps) unlike standard Levenshtein.
    Lower distance = more similar strings.
    """
    len1, len2 = len(s1), len(s2)
    
    # Create a dictionary to store the last occurrence of each character
    da = {}
    
    # First row and column (represent adding all letters from other string)
    max_dist = len1 + len2
    H = {}  # Dictionary for dynamic programming
    H[-1, -1] = max_dist
    
    for i in range(0, len1 + 1):
        H[i, -1] = max_dist
        H[i, 0] = i
    for j in range(0, len2 + 1):
        H[-1, j] = max_dist
        H[0, j] = j
    
    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = da.get(s2[j-1], 0)
            l = db
            if s1[i-1] == s2[j-1]:
                cost = 0
                db = j
            else:
                cost = 1
            H[i, j] = min(
                H[i-1, j] + 1,         # deletion
                H[i, j-1] + 1,         # insertion
                H[i-1, j-1] + cost,    # substitution
                H[k-1, l-1] + (i-k-1) + 1 + (j-l-1)  # transposition
            )
        da[s1[i-1]] = i
    
    return H[len1, len2]

# Initialize global n-gram model
print("Initializing n-gram model...")
ngram_model = NgramModel()

# Try to load from cache, otherwise train from dataset
if not ngram_model.load_cache():
    print("No cache found. Training from dataset...")
    ngram_model.train_from_dataset()
    ngram_model.save_cache()

# ---------------------------------------------------------------------
# DAMERAU-LEVENSHTEIN DISTANCE
# ---------------------------------------------------------------------
def damerau_levenshtein_distance(s1, s2):
    """
    Calculate Damerau-Levenshtein distance between two strings.
    This includes: insertions, deletions, substitutions, and transpositions.
    """
    len1, len2 = len(s1), len(s2)
    
    # Create distance matrix with extra row/column for empty string
    big_int = max(len1, len2) + 1
    H = {}
    
    # Initialize
    max_dist = len1 + len2
    H[-1, -1] = max_dist
    
    for i in range(0, len1 + 1):
        H[i, -1] = max_dist
        H[i, 0] = i
    for j in range(0, len2 + 1):
        H[-1, j] = max_dist
        H[0, j] = j
    
    for i in range(1, len1 + 1):
        DB = 0
        for j in range(1, len2 + 1):
            k = DB
            l = H.get((i - 1, j - 1), 0) if s1[i - 1] == s2[j - 1] else H.get((i - 1, j - 1), 0) + 1
            
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                DB = j
            else:
                cost = 1
            
            H[i, j] = min(
                H[i - 1, j] + 1,      # deletion
                H[i, j - 1] + 1,      # insertion
                H[i - 1, j - 1] + cost,  # substitution
            )
    
    return H[len1, len2]

def normalized_damerau_levenshtein(s1, s2):
    """
    Return normalized Damerau-Levenshtein distance (0-1 scale).
    0 = identical strings, 1 = completely different
    """
    distance = damerau_levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return distance / max_len

# ---------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------
def get_current_token(text):
    """Get the word currently being typed"""
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]

def get_context_words(text, n=2):
    """Get the last n words before current token for context"""
    words = text.strip().split()
    
    # If currently typing a word, exclude it from context
    if words and not text.endswith(" "):
        words = words[:-1]
    
    # Return last n words
    return words[-n:] if len(words) >= n else words

# ---------------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------------
class FilipinoIME(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("🇵🇭 Filipino Keyboard with N-gram Prediction")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Configure main grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create widgets
        self.create_widgets()
        
        # Bindings
        self.bind("<Tab>", lambda e: self.auto_complete())
        
        # Update suggestions initially
        self.update_suggestions()

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid
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
            text="🇵🇭 Filipino Keyboard with N-gram Prediction",
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
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=scrollbar.set)
        
        self.text_display.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.text_display.bind("<KeyRelease>", self.on_text_change)
        self.text_display.bind("<space>", lambda e: self.force_update_suggestions())
        
        # WORD COMPLETION LABEL
        completion_label = ttk.Label(
            main_container,
            text="Word Completion (N-gram based):",
            font=("Segoe UI", 14, "bold")
        )
        completion_label.grid(row=2, column=0, sticky="w", pady=(5, 5))
        
        # WORD COMPLETION CONTAINER
        self.completion_frame = ttk.Frame(main_container)
        self.completion_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.completion_frame.grid_columnconfigure(0, weight=1)
        
        self.completion_container = ttk.Frame(self.completion_frame)
        self.completion_container.grid(row=0, column=0, sticky="w")
        
        # PREDICTIVE LABEL
        predictive_label = ttk.Label(
            main_container,
            text="Next Word Prediction (N-gram based):",
            font=("Segoe UI", 14, "bold")
        )
        predictive_label.grid(row=4, column=0, sticky="w", pady=(5, 5))
        
        # PREDICTIVE CONTAINER
        self.predictive_frame = ttk.Frame(main_container)
        self.predictive_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.predictive_frame.grid_columnconfigure(0, weight=1)
        
        self.predictive_container = ttk.Frame(self.predictive_frame)
        self.predictive_container.grid(row=0, column=0, sticky="w")
        
        # KEYBOARD LABEL
        keyboard_label = ttk.Label(
            main_container,
            text="Virtual Keyboard:",
            font=("Segoe UI", 14, "bold")
        )
        keyboard_label.grid(row=6, column=0, sticky="w", pady=(5, 5))
        
        # KEYBOARD
        keyboard_frame = ttk.Frame(main_container)
        keyboard_frame.grid(row=7, column=0, sticky="nsew", pady=(0, 5))
        self.create_keyboard(keyboard_frame)
        
        # STATUS BAR
        self.status_bar = ttk.Label(main_container, text="Ready | N-gram model loaded", 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=8, column=0, sticky="ew", pady=(5, 0))

    def on_text_change(self, event=None):
        self.update_suggestions()
        current_text = self.text_display.get("1.0", "end-1c")
        word_count = len(current_text.split())
        self.status_bar.config(text=f"Words: {word_count} | Vocab size: {len(ngram_model.vocabulary)}")

    def force_update_suggestions(self):
        self.after(10, self.update_suggestions)

    def update_suggestions(self):
        """Update both completion and predictive suggestions using n-gram model"""
        # Clear existing buttons
        for widget in self.completion_container.winfo_children():
            widget.destroy()
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        # 1. WORD COMPLETION (when typing)
        if token:
            completions = ngram_model.get_completion_suggestions(token, context, max_results=8)
            
            if completions:
                for word in completions[:8]:
                    # Preserve capitalization of input
                    display_word = word.capitalize() if token and token[0].isupper() else word
                    
                    btn = ttk.Button(
                        self.completion_container,
                        text=display_word,
                        command=lambda w=display_word: self.apply_completion(w),
                        style="Completion.TButton"
                    )
                    btn.pack(side="left", padx=3, pady=3, ipadx=8, ipady=3)
            else:
                placeholder = ttk.Label(
                    self.completion_container,
                    text="No completions found",
                    font=("Segoe UI", 10),
                    foreground="gray"
                )
                placeholder.pack()
        else:
            placeholder = ttk.Label(
                self.completion_container,
                text="Start typing to see completions",
                font=("Segoe UI", 10, "italic"),
                foreground="gray"
            )
            placeholder.pack()
        
        # 2. NEXT WORD PREDICTION (when not typing)
        if not token or text.endswith(" "):
            predictions = ngram_model.get_next_word_suggestions(context, max_results=6)
            
            if predictions:
                for word in predictions:
                    # Capitalize if previous word was capitalized or start of sentence
                    should_capitalize = False
                    if not context:
                        should_capitalize = True
                    elif context and context[-1][0].isupper():
                        should_capitalize = True
                    
                    display_word = word.capitalize() if should_capitalize else word
                    
                    btn = ttk.Button(
                        self.predictive_container,
                        text=display_word,
                        command=lambda w=display_word: self.apply_prediction(w),
                        style="Predictive.TButton"
                    )
                    btn.pack(side="left", padx=5, pady=5, ipadx=12, ipady=5)
            else:
                placeholder = ttk.Label(
                    self.predictive_container,
                    text="No predictions available",
                    font=("Segoe UI", 10),
                    foreground="gray"
                )
                placeholder.pack()
        else:
            placeholder = ttk.Label(
                self.predictive_container,
                text="Finish current word for predictions",
                font=("Segoe UI", 10, "italic"),
                foreground="gray"
            )
            placeholder.pack()

    def apply_completion(self, word):
        """Apply word completion suggestion"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        
        if token:
            # Find position and replace
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            
            self.text_display.delete(start_index, "end-1c")
            self.text_display.insert(start_index, word + " ")
        else:
            self.text_display.insert("end", word + " ")
        
        self.update_suggestions()
        self.status_bar.config(text=f"Applied: '{word}'")
    
    def apply_prediction(self, word):
        """Apply next word prediction"""
        text = self.text_display.get("1.0", "end-1c")
        
        # Add space if needed
        if text and not text.endswith(" "):
            self.text_display.insert("end", " ")
        
        self.text_display.insert("end", word + " ")
        self.update_suggestions()
        self.status_bar.config(text=f"Predicted: '{word}'")

    def auto_complete(self):
        """Auto-complete using top suggestion"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        if token and len(token) >= 2:
            completions = ngram_model.get_completion_suggestions(token, context, max_results=1)
            if completions:
                self.apply_completion(completions[0])
                return "break"  # Prevent default tab behavior

    def insert_char(self, char):
        self.text_display.insert("insert", char)
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()  # Trigger suggestions after virtual keyboard input

    def backspace(self):
        self.text_display.delete("insert-1c")
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()  # Trigger suggestions after backspace
    
    def space(self):
        self.text_display.insert("insert", " ")
        self.text_display.event_generate("<KeyRelease>")
        self.force_update_suggestions()  # Trigger next-word predictions after space
    
    def enter(self):
        self.text_display.insert("insert", "\n")
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()  # Trigger suggestions after enter

    def create_keyboard(self, parent):
        # Configure styles
        style = ttk.Style()
        style.configure("Keyboard.TButton", font=("Segoe UI", 12), padding=6)
        style.configure("Predictive.TButton", font=("Segoe UI", 11), padding=8)
        style.configure("Completion.TButton", font=("Segoe UI", 10), padding=4)
        
        parent.grid_columnconfigure(0, weight=1)
        
        # Function row
        func_row = ttk.Frame(parent)
        func_row.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        auto_btn = ttk.Button(func_row, text="TAB: Auto-Complete", 
                            style="Keyboard.TButton",
                            command=self.auto_complete)
        auto_btn.pack(side="left", padx=2, ipadx=5, expand=True, fill="x")
        
        clear_btn = ttk.Button(func_row, text="Clear Text", 
                             style="Keyboard.TButton",
                             command=lambda: [self.text_display.delete("1.0", "end"), 
                                            self.update_suggestions()])
        clear_btn.pack(side="left", padx=2, ipadx=5, expand=True, fill="x")
        
        # Number row
        row_num = ttk.Frame(parent)
        row_num.grid(row=1, column=0, sticky="ew", pady=2)
        
        for ch in "1234567890":
            btn = ttk.Button(row_num, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        backspace_btn = ttk.Button(row_num, text="⌫", 
                                  style="Keyboard.TButton",
                                  command=self.backspace)
        backspace_btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # Letter rows
        row1 = ttk.Frame(parent)
        row1.grid(row=2, column=0, sticky="ew", pady=2)
        for ch in "qwertyuiop":
            btn = ttk.Button(row1, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        row2 = ttk.Frame(parent)
        row2.grid(row=3, column=0, sticky="ew", pady=2)
        ttk.Frame(row2, width=20).pack(side="left")
        for ch in "asdfghjkl":
            btn = ttk.Button(row2, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        enter_btn = ttk.Button(row2, text="↵", 
                              style="Keyboard.TButton",
                              command=self.enter)
        enter_btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        row3 = ttk.Frame(parent)
        row3.grid(row=4, column=0, sticky="ew", pady=2)
        ttk.Frame(row3, width=40).pack(side="left")
        for ch in "zxcvbnm":
            btn = ttk.Button(row3, text=ch, style="Keyboard.TButton",
                           command=lambda c=ch: self.insert_char(c))
            btn.pack(side="left", padx=1, ipadx=5, expand=True, fill="x")
        
        # Space bar
        space_row = ttk.Frame(parent)
        space_row.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        space_btn = ttk.Button(space_row, text="SPACE", 
                              style="Keyboard.TButton",
                              command=self.space)
        space_btn.pack(ipadx=50, ipady=10, expand=True, fill="x")

# ---------------------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Filipino Keyboard with N-gram Prediction")
    print("="*60)
    app = FilipinoIME()
    app.mainloop()
