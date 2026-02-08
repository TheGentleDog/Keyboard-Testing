import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
from datasets import load_dataset
import pickle

# Configuration
MAX_SUGGESTIONS = 5
NGRAM_CACHE_FILE = "ngram_model.pkl"
VOCAB_CACHE_FILE = "vocabulary.pkl"
USER_LEARNING_FILE = "user_learning.json"  # NEW: Store user-specific learning

# GitHub shortcut library URLs
SHORTCUT_LIBRARY_URLS = [
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/train_words.csv",
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/test_words.csv"
]
SHORTCUT_LIBRARY_CACHE = "shortcut_library.json"

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
        
        # NEW: Character/syllable-level n-grams for within-word completion
        self.char_bigrams = defaultdict(Counter)   # "ta" → {"yo": 50, "ga": 30}
        self.char_trigrams = defaultdict(Counter)  # ("t", "a") → {"y": 50, "l": 30}
        
        # NEW: User learning tracking
        self.user_shortcuts = {}  # User-typed shortcuts → full words
        self.user_shortcut_usage = Counter()  # How often user uses each shortcut
        self.manual_shortcuts = {}  # Manually added shortcuts
        self.new_words = set()  # Words user added that aren't in dataset
        self.word_usage_history = []  # Track user's typing patterns
        
        # NEW: CSV shortcut library (from GitHub)
        self.csv_shortcuts = {}  # Loaded from CSV files (informal→formal)
        
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
                    
                    # Build character-level n-grams within each word
                    self._build_char_ngrams(token)
                    
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
    
    def _build_char_ngrams(self, word):
        """
        Build character-level bigrams and trigrams within a word.
        Example: "tayo" → bigrams: "ta"→"y", "ay"→"o", "yo"→END
                        trigrams: ("t","a")→"y", ("a","y")→"o"
        """
        if len(word) < 2:
            return
        
        # Character bigrams (2-char sequences)
        for i in range(len(word) - 1):
            bigram = word[i:i+2]  # "ta", "ay", "yo"
            if i < len(word) - 2:
                next_char = word[i+2]
                self.char_bigrams[bigram][next_char] += 1
            else:
                # Mark end of word
                self.char_bigrams[bigram]["<END>"] += 1
        
        # Character trigrams (3-char sequences predicting 4th)
        for i in range(len(word) - 2):
            trigram = (word[i], word[i+1])
            if i < len(word) - 3:
                next_char = word[i+3]
                self.char_trigrams[trigram][next_char] += 1
    
    def get_char_level_completions(self, prefix, max_results=5):
        """
        Get word completions based on character-level n-grams.
        
        Example: prefix="ta" 
        → Look up what commonly follows "ta" in Filipino words
        → Returns: ["tayo", "talaga", "tahimik", ...]
        """
        if len(prefix) < 2:
            return []
        
        completions = []
        
        # Minimum word length to avoid single letters
        min_length = max(len(prefix) + 1, 3)  # At least 3 chars or prefix+1
        
        # Find words that start with this prefix and meet minimum length
        candidates = [
            word for word in self.vocabulary 
            if word.startswith(prefix) and len(word) >= min_length
        ]
        
        # Score based on character continuation probability
        scored = []
        for word in candidates:
            score = 0
            
            # Check character bigram patterns
            for i in range(len(prefix), len(word) - 1):
                if i >= 2:
                    bigram = word[i-2:i]
                    if bigram in self.char_bigrams:
                        next_char = word[i]
                        count = self.char_bigrams[bigram].get(next_char, 0)
                        total = sum(self.char_bigrams[bigram].values())
                        if total > 0:
                            score += count / total
            
            # Boost score based on word frequency
            freq_score = self.unigrams.get(word, 0) / max(self.total_words, 1)
            final_score = score + (freq_score * 10)  # Weight word frequency higher
            
            scored.append((word, final_score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in scored[:max_results]]
    
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
            """Replace common syllables with numbers"""
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
    
    def load_shortcut_library(self):
        """
        Load additional shortcuts from GitHub CSV files.
        These contain informal→formal Filipino word mappings.
        
        CSV format: informal,formal
        Example: "lng,lang"
        """
        # Try to load from cache first
        if os.path.exists(SHORTCUT_LIBRARY_CACHE):
            try:
                with open(SHORTCUT_LIBRARY_CACHE, 'r', encoding='utf-8') as f:
                    self.csv_shortcuts = json.load(f)
                print(f"✓ Loaded {len(self.csv_shortcuts)} shortcuts from cache")
                return
            except Exception as e:
                print(f"⚠ Error loading cached shortcuts: {e}")
        
        # Download and parse CSV files
        print("\n📚 Loading shortcut library from GitHub...")
        
        import urllib.request
        import csv
        import io
        
        for url in SHORTCUT_LIBRARY_URLS:
            try:
                print(f"  Downloading: {url.split('/')[-1]}...")
                
                # Download CSV
                response = urllib.request.urlopen(url, timeout=10)
                csv_data = response.read().decode('utf-8')
                
                # Parse CSV
                csv_reader = csv.reader(io.StringIO(csv_data))
                next(csv_reader, None)  # Skip header if exists
                
                count = 0
                for row in csv_reader:
                    if len(row) >= 2:
                        informal = row[0].strip().lower()
                        formal = row[1].strip().lower()
                        
                        # Only add if informal is shorter (it's a shortcut)
                        if informal and formal and len(informal) < len(formal):
                            self.csv_shortcuts[informal] = formal
                            count += 1
                
                print(f"  ✓ Loaded {count} shortcuts from {url.split('/')[-1]}")
                
            except Exception as e:
                print(f"  ⚠ Error loading {url.split('/')[-1]}: {e}")
        
        # Save to cache
        try:
            with open(SHORTCUT_LIBRARY_CACHE, 'w', encoding='utf-8') as f:
                json.dump(self.csv_shortcuts, f, indent=2, ensure_ascii=False)
            print(f"✓ Cached {len(self.csv_shortcuts)} shortcuts")
        except Exception as e:
            print(f"⚠ Error caching shortcuts: {e}")
    
    def resolve_shortcut(self, word):
        """
        Resolve a word to its full form if it's a shortcut.
        Priority: User > CSV Library > Dataset
        """
        word_lower = word.lower()
        
        # Priority 1: User shortcuts (highest - user's personal patterns)
        if word_lower in self.user_shortcuts:
            return self.user_shortcuts[word_lower]
        
        # Priority 2: CSV shortcut library (curated informal→formal)
        if word_lower in self.csv_shortcuts:
            return self.csv_shortcuts[word_lower]
        
        # Priority 3: Dataset-learned shortcuts
        if word_lower in self.learned_shortcuts:
            return self.learned_shortcuts[word_lower]
        
        return word
    
    def get_all_shortcut_expansions(self, shortcut):
        """
        Get ALL possible expansions of a shortcut.
        Returns list of (word, source) tuples.
        Used for multiple expansion suggestions.
        """
        shortcut_lower = shortcut.lower()
        expansions = []
        
        # Collect from all sources
        if shortcut_lower in self.user_shortcuts:
            expansions.append((self.user_shortcuts[shortcut_lower], 'user'))
        
        if shortcut_lower in self.csv_shortcuts:
            word = self.csv_shortcuts[shortcut_lower]
            if word not in [w for w, _ in expansions]:
                expansions.append((word, 'csv'))
        
        if shortcut_lower in self.learned_shortcuts:
            word = self.learned_shortcuts[shortcut_lower]
            if word not in [w for w, _ in expansions]:
                expansions.append((word, 'learned'))
        
        return expansions
    
    def learn_from_user_typing(self, typed_shortcut, selected_word):
        """
        Learn a shortcut pattern from user behavior.
        Called when user types a shortcut then selects a word.
        
        Example: User types "lng" → selects "lang" → Learn "lng" → "lang"
        """
        typed_shortcut = typed_shortcut.lower()
        selected_word = selected_word.lower()
        
        # Only learn if shortcut is significantly shorter
        if len(typed_shortcut) < len(selected_word) - 1:
            # Track this pattern
            if typed_shortcut not in self.user_shortcuts:
                self.user_shortcuts[typed_shortcut] = selected_word
                print(f"🎓 Learned from typing: '{typed_shortcut}' → '{selected_word}'")
            
            # Track usage frequency
            self.user_shortcut_usage[typed_shortcut] += 1
            
            # Save to disk
            self.save_user_learning()
    
    def add_new_word(self, word):
        """
        Add a new word to vocabulary that wasn't in the dataset.
        """
        word_lower = word.lower()
        if word_lower not in self.vocabulary:
            self.vocabulary.add(word_lower)
            self.new_words.add(word_lower)
            self.unigrams[word_lower] = 1  # Initialize with count of 1
            print(f"📝 Added new word to vocabulary: '{word}'")
            self.save_user_learning()
    
    def track_word_usage(self, word, context=None):
        """
        Track user's word usage to improve predictions over time.
        Updates n-gram counts based on actual usage.
        """
        word_lower = word.lower()
        
        # Update unigram count
        self.unigrams[word_lower] += 1
        self.total_words += 1
        
        # Update bigram if context available
        if context and len(context) >= 1:
            prev = context[-1].lower()
            prev = self.resolve_shortcut(prev)
            self.bigrams[prev][word_lower] += 1
        
        # Update trigram if context available
        if context and len(context) >= 2:
            prev2 = context[-2].lower()
            prev1 = context[-1].lower()
            prev2 = self.resolve_shortcut(prev2)
            prev1 = self.resolve_shortcut(prev1)
            self.trigrams[(prev2, prev1)][word_lower] += 1
        
        # Add to history (keep last 1000 words)
        self.word_usage_history.append((word_lower, context))
        if len(self.word_usage_history) > 1000:
            self.word_usage_history.pop(0)
    
    def save_user_learning(self):
        """
        Save user-specific learning data to JSON file.
        Separate from main model cache for easier updates.
        """
        user_data = {
            'user_shortcuts': self.user_shortcuts,
            'user_shortcut_usage': dict(self.user_shortcut_usage),
            'new_words': list(self.new_words),
            'word_usage_history': self.word_usage_history[-100:]  # Keep last 100
        }
        
        try:
            with open(USER_LEARNING_FILE, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Error saving user learning: {e}")
    
    def load_user_learning(self):
        """
        Load user-specific learning data from JSON file.
        """
        if not os.path.exists(USER_LEARNING_FILE):
            print("ℹ No user learning file found (will be created on first use)")
            return
        
        try:
            with open(USER_LEARNING_FILE, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            self.user_shortcuts = user_data.get('user_shortcuts', {})
            self.user_shortcut_usage = Counter(user_data.get('user_shortcut_usage', {}))
            self.new_words = set(user_data.get('new_words', []))
            self.word_usage_history = user_data.get('word_usage_history', [])
            
            # Add new words back to vocabulary
            self.vocabulary.update(self.new_words)
            
            print(f"✓ Loaded user learning:")
            print(f"  User shortcuts: {len(self.user_shortcuts)}")
            print(f"  New words: {len(self.new_words)}")
            
        except Exception as e:
            print(f"⚠ Error loading user learning: {e}")
    
    def save_cache(self):
        """Save trained model to disk"""
        data = {
            'unigrams': dict(self.unigrams),
            'bigrams': {k: dict(v) for k, v in self.bigrams.items()},
            'trigrams': {k: dict(v) for k, v in self.trigrams.items()},
            'char_bigrams': {k: dict(v) for k, v in self.char_bigrams.items()},
            'char_trigrams': {k: dict(v) for k, v in self.char_trigrams.items()},
            'vocabulary': list(self.vocabulary),
            'total_words': self.total_words,
            'learned_shortcuts': self.learned_shortcuts,
            'shortcut_frequencies': dict(self.shortcut_frequencies)
        }
        with open(NGRAM_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved n-gram model to {NGRAM_CACHE_FILE}")
        print(f"✓ Saved {len(self.learned_shortcuts)} learned shortcuts")
        print(f"✓ Saved {len(self.char_bigrams)} character bigrams")
    
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
            
            # Load character-level n-grams if available
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
            
            # Load learned shortcuts if available
            self.learned_shortcuts = data.get('learned_shortcuts', {})
            self.shortcut_frequencies = Counter(data.get('shortcut_frequencies', {}))
            
            print(f"✓ Loaded cached n-gram model")
            print(f"  Vocabulary: {len(self.vocabulary)} words")
            print(f"  Total words: {self.total_words}")
            print(f"  Learned shortcuts: {len(self.learned_shortcuts)}")
            print(f"  Character bigrams: {len(self.char_bigrams)}")
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
        
        # PRIORITY 0: Check ALL shortcut sources for multiple expansions
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if all_expansions:
            # Add all possible expansions with priority based on source
            for full_word, source in all_expansions:
                # Priority: User (3x) > CSV Library (2x) > Dataset (1.5x)
                if source == 'user':
                    priority_multiplier = 3.0
                elif source == 'csv':
                    priority_multiplier = 2.0
                else:  # learned
                    priority_multiplier = 1.5
                
                shortcut_candidates.append((full_word, True, True, priority_multiplier))  # word, exact, shortcut, priority
        
        # Find all words that start with prefix OR are close via edit distance
        candidates = []
        
        # Strategy 1: Exact prefix matches (best candidates)
        # Filter out very short words unless prefix is also very short
        min_word_length = max(len(prefix) + 1, 3)  # At least 3 chars, or prefix+1
        
        exact_matches = [
            word for word in self.vocabulary 
            if word.startswith(prefix) and len(word) >= min_word_length
        ]
        
        # Strategy 1.5: CHARACTER-LEVEL N-GRAM boost (NEW!)
        # Get character-based completions and boost their scores
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        
        for word in exact_matches:
            # Skip single character or very short words unless prefix is also very short
            if len(word) < 2 and len(prefix) < 2:
                continue
            
            # Check if word is in character-level completions (has good char patterns)
            if word in char_completions:
                # Higher priority - good character continuation patterns
                candidates.append((word, True, False, 1.5))  # 1.5x boost for char patterns
            else:
                candidates.append((word, True, False, 1.0))  # Normal priority
        
        # Add character completions that might not be in exact matches
        for word in char_completions:
            if word not in exact_matches and len(word) >= min_word_length:
                candidates.append((word, True, False, 1.3))  # Still boost, but less
        
        # Strategy 2: Typo tolerance - find words with small edit distance
        # Only if we have few exact matches
        if len(exact_matches) < max_results:
            for word in self.vocabulary:
                if word not in exact_matches:
                    # Skip very short words
                    if len(word) < min_word_length:
                        continue
                    
                    # Only consider words of similar length (within 3 chars for better flexibility)
                    if abs(len(word) - len(prefix)) <= 3:
                        distance = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                        # Only include if distance is small (1-2 edits)
                        if distance <= 2:
                            candidates.append((word, False, False, 1.0))  # Not exact, not shortcut, default priority
        
        # Combine shortcut candidates with regular candidates
        all_candidates = shortcut_candidates + candidates
        
        # Score each candidate
        scored = []
        for word, is_exact_match, is_shortcut, priority_mult in all_candidates:
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
            
            # 5. SHORTCUT BONUS with priority multiplier!
            # User shortcuts get 3x, manual get 2x, learned get 1.5x
            shortcut_bonus = 5.0 * priority_mult if is_shortcut else 1.0
            
            # 6. Track usage frequency bonus (from user_shortcut_usage)
            usage_bonus = 1.0
            if is_shortcut and prefix in self.user_shortcut_usage:
                # More used shortcuts get higher priority
                usage_count = self.user_shortcut_usage[prefix]
                usage_bonus = 1.0 + min(usage_count / 10.0, 2.0)  # Cap at 3x
            
            # Combined score with weighted factors
            # PRIMARY: N-gram probability (70%) - language model comes first
            # SECONDARY: Damerau-Levenshtein similarity (20%) - typo tolerance
            # TERTIARY: Prefix ratio (10%) - completion quality
            final_score = (
                prob * 0.7 +           # 70% n-gram probability (PRIMARY)
                dl_score * 0.2 +       # 20% edit distance similarity (SECONDARY)
                prefix_ratio * 0.1     # 10% prefix coverage (TERTIARY)
            ) * exact_bonus * shortcut_bonus * usage_bonus  # All bonuses multiply
            
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

# Load user-specific learning (shortcuts, new words, etc.)
print("\nLoading user learning...")
ngram_model.load_user_learning()

# Load additional shortcut library from GitHub CSV files
print("\nLoading shortcut library...")
ngram_model.load_shortcut_library()

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
        context = get_context_words(text, n=2)
        
        if token:
            # LEARN FROM USER: Track shortcut pattern
            ngram_model.learn_from_user_typing(token, word)
            
            # Find position and replace
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            
            self.text_display.delete(start_index, "end-1c")
            self.text_display.insert(start_index, word + " ")
        else:
            self.text_display.insert("end", word + " ")
        
        # LEARN FROM USER: Track word usage
        ngram_model.track_word_usage(word, context)
        
        self.update_suggestions()
        self.status_bar.config(text=f"Applied: '{word}'")
    
    def apply_prediction(self, word):
        """Apply next word prediction"""
        text = self.text_display.get("1.0", "end-1c")
        context = get_context_words(text, n=2)
        
        # Add space if needed
        if text and not text.endswith(" "):
            self.text_display.insert("end", " ")
        
        self.text_display.insert("end", word + " ")
        
        # LEARN FROM USER: Track word usage with context
        ngram_model.track_word_usage(word, context)
        
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
