import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
from datasets import load_dataset
import pickle

# Configuration
MAX_SUGGESTIONS = 8
NGRAM_CACHE_FILE = "ngram_model_conversational.pkl"
VOCAB_CACHE_FILE = "vocabulary_conversational.pkl"
USER_LEARNING_FILE = "user_learning.json"

# GitHub shortcut library URLs (Filipino informal→formal)
SHORTCUT_LIBRARY_URLS = [
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/train_words.csv",
    "https://raw.githubusercontent.com/ljyflores/efficient-spelling-normalization-filipino/main/data/test_words.csv"
]
SHORTCUT_LIBRARY_CACHE = "shortcut_library.json"

# NEW: Common conversational shortcuts for English
ENGLISH_SHORTCUTS = {
    # Common texting abbreviations
    'u': 'you', 'ur': 'your', 'ure': "you're", 'r': 'are',
    'n': 'and', 'w': 'with', 'b4': 'before', 'l8r': 'later',
    'plz': 'please', 'pls': 'please', 'thx': 'thanks', 'ty': 'thank you',
    'brb': 'be right back', 'btw': 'by the way', 'omg': 'oh my god',
    'lol': 'laughing out loud', 'idk': "I don't know", 'tbh': 'to be honest',
    'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'fyi': 'for your information', 'asap': 'as soon as possible',
    'pov': 'point of view', 'rn': 'right now', 'nvm': 'never mind',
    'msg': 'message', 'txt': 'text', 'pic': 'picture',
    'lmk': 'let me know', 'irl': 'in real life', 'tho': 'though',
    'cuz': 'because', 'coz': 'because', 'gonna': 'going to',
    'wanna': 'want to', 'gotta': 'got to', 'kinda': 'kind of',
    'sorta': 'sort of', 'dunno': "don't know", 'yolo': 'you only live once',
    # Emoticon/emoji shortcuts
    ':)': '🙂', ':(': '☹️', ':D': '😃', ';)': '😉',
    '<3': '❤️', 'xd': '😆', 'lmao': '😂'
}

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
        self.learned_shortcuts = {}
        self.shortcut_frequencies = Counter()
        
        # Character/syllable-level n-grams for within-word completion
        self.char_bigrams = defaultdict(Counter)
        self.char_trigrams = defaultdict(Counter)
        
        # User learning tracking
        self.user_shortcuts = {}
        self.user_shortcut_usage = Counter()
        self.manual_shortcuts = {}
        self.new_words = set()
        self.word_usage_history = []
        
        # CSV shortcut library (from GitHub)
        self.csv_shortcuts = {}
        
        # NEW: Track which language each word belongs to
        self.language_tags = {}  # word -> 'fil' or 'eng' or 'both'
        
    def train_from_conversational_datasets(self, languages=['filipino', 'english']):
        """
        Load and train from conversational datasets for better chat predictions.
        
        Args:
            languages: List containing 'filipino' and/or 'english'
        """
        print("\n" + "="*60)
        print("LOADING CONVERSATIONAL DATASETS")
        print("="*60)
        
        for lang in languages:
            if lang.lower() == 'filipino':
                self._load_filipino_conversational()
            elif lang.lower() == 'english':
                self._load_english_conversational()
        
        print(f"\n✓ Final vocabulary: {len(self.vocabulary)} unique words")
        print(f"✓ Total words processed: {self.total_words}")
        print(f"✓ Bigram contexts: {len(self.bigrams)}")
        print(f"✓ Trigram contexts: {len(self.trigrams)}")
        
        # Learn shortcuts from vocabulary
        print("\n🔍 Learning shortcuts from dataset...")
        self._learn_shortcuts_from_vocabulary()
    
    def _load_filipino_conversational(self):
        """Load Filipino conversational datasets (Tagalog only)"""
        print("\n📱 Loading Filipino/Tagalog conversational data...")
        
        success = False
        
        # Strategy 1: Try jcblaise Filipino datasets (text benchmarks)
        try:
            print("  Attempting jcblaise/hatespeech dataset...")
            dataset = load_dataset("jcblaise/hatespeech", split="train[:5000]")
            
            for example in dataset:
                if 'text' in example:
                    text = example['text']
                    tokens = self._clean_and_tokenize(text)
                    self._process_tokens(tokens, language='filipino')
            
            print(f"  ✓ Processed hatespeech: {len(self.vocabulary)} words so far")
            success = True
            
        except Exception as e:
            print(f"  ⚠ Error loading jcblaise/hatespeech: {e}")
        
        # Strategy 2: Try another jcblaise dataset
        if not success:
            try:
                print("  Attempting jcblaise/dengue dataset...")
                dataset = load_dataset("jcblaise/dengue", split="train")
                
                for example in dataset:
                    if 'text' in example:
                        text = example['text']
                        tokens = self._clean_and_tokenize(text)
                        self._process_tokens(tokens, language='filipino')
                
                print(f"  ✓ Processed dengue: {len(self.vocabulary)} words so far")
                success = True
                
            except Exception as e:
                print(f"  ⚠ Error loading dengue: {e}")
        
        # Strategy 3: Try Filipino sentiment dataset
        if not success:
            try:
                print("  Attempting Filipino sentiment dataset...")
                dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "filipino", split="train[:3000]")
                
                for example in dataset:
                    if 'text' in example:
                        tokens = self._clean_and_tokenize(example['text'])
                        self._process_tokens(tokens, language='filipino')
                
                print(f"  ✓ Processed sentiment dataset: {len(self.vocabulary)} words so far")
                success = True
                
            except Exception as e:
                print(f"  ⚠ Error loading sentiment dataset: {e}")
        
        # If all fail, use fallback
        if not success:
            print("  All datasets failed, using fallback vocabulary...")
            self._load_filipino_fallback()
    
    def _load_english_conversational(self):
        """Load English conversational datasets"""
        print("\n💬 Loading English conversational data...")
        
        success = False
        
        # Strategy 1: Try DialogSum (proven to work, has real conversations)
        try:
            print("  Attempting DialogSum dataset...")
            dataset = load_dataset("knkarthick/dialogsum", split="train[:3000]")
            
            for example in dataset:
                # DialogSum has 'dialogue' field with conversations
                if 'dialogue' in example:
                    # Parse the dialogue (format: #Person1#: text\n#Person2#: text)
                    dialogue_text = example['dialogue']
                    # Split by speaker and process each utterance
                    for line in dialogue_text.split('\n'):
                        if ':' in line:
                            # Remove speaker tag and process text
                            text = line.split(':', 1)[1].strip()
                            tokens = self._clean_and_tokenize(text)
                            self._process_tokens(tokens, language='english')
            
            print(f"  ✓ Processed DialogSum: {len(self.vocabulary)} words so far")
            success = True
            
        except Exception as e:
            print(f"  ⚠ Error loading DialogSum: {e}")
        
        # Strategy 2: Try everyday conversations dataset
        if not success:
            try:
                print("  Attempting everyday conversations dataset...")
                dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train[:2000]")
                
                for example in dataset:
                    # This dataset has 'messages' field with conversation turns
                    if 'messages' in example:
                        for message in example['messages']:
                            if 'content' in message:
                                tokens = self._clean_and_tokenize(message['content'])
                                self._process_tokens(tokens, language='english')
                
                print(f"  ✓ Processed everyday conversations: {len(self.vocabulary)} words so far")
                success = True
                
            except Exception as e:
                print(f"  ⚠ Error loading everyday conversations: {e}")
        
        # Strategy 3: Try a simple text dataset
        if not success:
            try:
                print("  Attempting openwebtext subset...")
                dataset = load_dataset("stas/openwebtext-10k", split="train[:1000]")
                
                for example in dataset:
                    if 'text' in example:
                        # Take only first few sentences to avoid overwhelming
                        text = example['text'][:500]
                        tokens = self._clean_and_tokenize(text)
                        self._process_tokens(tokens, language='english')
                
                print(f"  ✓ Processed openwebtext: {len(self.vocabulary)} words so far")
                success = True
                
            except Exception as e:
                print(f"  ⚠ Error loading openwebtext: {e}")
        
        # If all fail, use fallback
        if not success:
            print("  All datasets failed, using fallback vocabulary...")
            self._load_english_fallback()
    
    def _clean_and_tokenize(self, text):
        """
        Clean and tokenize text for conversational context.
        Preserves common chat elements like emoticons.
        """
        if not text:
            return []
        
        # Basic cleaning - preserve emoticons and common punctuation
        text = text.strip()
        
        # Simple whitespace tokenization
        # More sophisticated tokenization could be added here
        tokens = text.lower().split()
        
        # Filter out very long tokens (likely URLs or garbage)
        tokens = [t for t in tokens if len(t) < 20 and t.strip()]
        
        # Remove pure punctuation tokens (but keep emoticons)
        cleaned = []
        for token in tokens:
            # Keep if it has at least one alphanumeric character OR is an emoticon
            if any(c.isalnum() for c in token) or token in [':)', ':(', ':D', ';)', '<3']:
                cleaned.append(token.strip('.,!?;:'))
        
        return cleaned
    
    def _process_tokens(self, tokens, language='filipino'):
        """
        Process a list of tokens and update n-gram counts.
        
        Args:
            tokens: List of word tokens
            language: 'filipino' or 'english' for language tagging
        """
        if not tokens:
            return
        
        # Update vocabulary and tag language
        for token in tokens:
            self.vocabulary.add(token)
            
            # Tag language (if word appears in both, tag as 'both')
            if token in self.language_tags:
                if self.language_tags[token] != language:
                    self.language_tags[token] = 'both'
            else:
                self.language_tags[token] = language
        
        # Build n-grams
        for i, token in enumerate(tokens):
            # Unigram
            self.unigrams[token] += 1
            self.total_words += 1
            
            # Build character-level n-grams within each word
            self._build_char_ngrams(token)
            
            # Bigram
            if i > 0:
                prev = tokens[i-1]
                self.bigrams[prev][token] += 1
            
            # Trigram
            if i > 1:
                prev2 = tokens[i-2]
                prev1 = tokens[i-1]
                context = (prev2, prev1)
                self.trigrams[context][token] += 1
    
    def _load_filipino_fallback(self):
        """Enhanced Filipino fallback with conversational words"""
        fallback = [
            # Common function words
            "ang", "ng", "sa", "mga", "na", "ay", "at", "si", "ni", "kay",
            # Pronouns
            "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
            # Demonstratives
            "ito", "iyan", "iyon", "dito", "doon", "diyan",
            # Common verbs
            "kumain", "uminom", "matulog", "maglaro", "magbasa", "magsulat",
            # Conversational words
            "oo", "hindi", "sige", "talaga", "naman", "lang", "kasi",
            "bakit", "paano", "kailan", "saan", "ano", "sino",
            "pwede", "gusto", "ayaw", "kailangan", "dapat",
            # Adjectives
            "maganda", "pangit", "mabuti", "masama", "malaki", "maliit",
            # Nouns
            "bahay", "eskwela", "trabaho", "pamilya", "kaibigan", "pagkain",
            # Chat words
            "chat", "text", "message", "tanong", "sagot"
        ]
        
        for word in fallback:
            self.vocabulary.add(word)
            self.unigrams[word] = 10
            self.language_tags[word] = 'filipino'
        
        self.total_words += len(fallback) * 10
        print(f"  ✓ Loaded {len(fallback)} Filipino fallback words")
    
    def _load_english_fallback(self):
        """Enhanced English fallback with conversational words"""
        fallback = [
            # Common words
            "the", "a", "an", "and", "or", "but", "if", "when", "where",
            "what", "who", "why", "how", "can", "could", "would", "should",
            # Pronouns
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            # Verbs
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "go", "going", "went", "come", "coming", "came",
            # Chat words
            "yeah", "yes", "no", "ok", "okay", "sure", "maybe",
            "think", "know", "like", "want", "need", "see",
            "tell", "say", "ask", "talk", "chat", "text",
            # Conversational
            "really", "actually", "basically", "literally", "totally",
            "pretty", "very", "so", "too", "just", "only",
            "hello", "hi", "hey", "bye", "thanks", "please", "sorry"
        ]
        
        for word in fallback:
            word_lower = word.lower()
            self.vocabulary.add(word_lower)
            self.unigrams[word_lower] = 10
            self.language_tags[word_lower] = 'english'
        
        self.total_words += len(fallback) * 10
        print(f"  ✓ Loaded {len(fallback)} English fallback words")
    
    def _build_char_ngrams(self, word):
        """Build character-level bigrams and trigrams within a word"""
        if len(word) < 2:
            return
        
        # Character bigrams
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            if i < len(word) - 2:
                next_char = word[i+2]
                self.char_bigrams[bigram][next_char] += 1
            else:
                self.char_bigrams[bigram]["<END>"] += 1
        
        # Character trigrams
        for i in range(len(word) - 2):
            trigram = (word[i], word[i+1])
            if i < len(word) - 3:
                next_char = word[i+3]
                self.char_trigrams[trigram][next_char] += 1
    
    def get_char_level_completions(self, prefix, max_results=5):
        """Get word completions based on character-level n-grams"""
        if len(prefix) < 1:
            return []
        
        completions = []
        
        # Minimum word length based on prefix
        if len(prefix) == 1:
            min_length = 2
        else:
            min_length = max(len(prefix) + 1, 3)
        
        # Find words that start with prefix
        candidates = [
            word for word in self.vocabulary 
            if word.startswith(prefix) 
            and len(word) >= min_length
            and word.isalpha()
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
            
            # Boost based on word frequency
            freq_score = self.unigrams.get(word, 0) / max(self.total_words, 1)
            final_score = score + (freq_score * 10)
            
            scored.append((word, final_score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in scored[:max_results]]
    
    def _learn_shortcuts_from_vocabulary(self):
        """Learn common shortcut patterns from vocabulary"""
        def remove_vowels(word):
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
            if freq < 3:  # Lower threshold for conversational data
                continue
            
            shortcuts = []
            
            # Remove vowels
            vowel_removed = remove_vowels(word)
            if vowel_removed and len(vowel_removed) < len(word) and len(vowel_removed) >= 2:
                shortcuts.append(vowel_removed)
            
            # Number substitution
            num_shortcut = create_number_shortcut(word)
            if num_shortcut and num_shortcut != word:
                shortcuts.append(num_shortcut)
            
            # First chars (for longer words)
            if len(word) >= 4:
                shortcuts.append(word[:3])
                shortcuts.append(word[:2])
            
            # Add shortcuts
            for shortcut in shortcuts:
                if len(shortcut) < len(word):
                    shortcut_candidates[shortcut.lower()].append((word, freq))
        
        print(f"  Found {len(shortcut_candidates)} potential patterns")
        
        # Filter and select best shortcuts
        for shortcut, word_list in shortcut_candidates.items():
            if len(shortcut) < 2:
                continue
            
            word_list.sort(key=lambda x: x[1], reverse=True)
            
            # Accept shortcuts
            if len(word_list) == 1:
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
            elif len(word_list) > 1 and word_list[0][1] >= word_list[1][1] * 2:
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
            elif len(word_list) > 1 and word_list[0][1] > 20:
                self.learned_shortcuts[shortcut] = word_list[0][0]
                self.shortcut_frequencies[shortcut] = word_list[0][1]
        
        print(f"✓ Learned {len(self.learned_shortcuts)} shortcuts from dataset")
        
        # Add Filipino shortcuts manually
        filipino_shortcuts = {
            'lng': 'lang', 'nmn': 'naman', 'ksi': 'kasi', 'kng': 'kung',
            'd2': 'dito', 'dn': 'doon', 'dyn': 'diyan',
            'pde': 'pwede', 'pwd': 'pwede', 'sna': 'sana', 'tlg': 'talaga',
            'sya': 'siya', 'xa': 'siya', 'aq': 'ako', 'u': 'ikaw',
            'nde': 'hindi', 'hnd': 'hindi', 'pra': 'para',
            'wla': 'wala', 'my': 'may', 'ung': 'yung', 'un': 'yun',
            'q': 'ako', 'aq': 'ako', 'aqh': 'ako', 'k': 'ok',
            'mg': 'maga', 'kht': 'kahit', 'khit': 'kahit'
        }
        
        # Add English shortcuts
        english_shortcuts = ENGLISH_SHORTCUTS.copy()
        
        # Merge all manual shortcuts
        all_manual = {**filipino_shortcuts, **english_shortcuts}
        
        added_manual = 0
        for shortcut, full_word in all_manual.items():
            if shortcut not in self.learned_shortcuts:
                self.learned_shortcuts[shortcut] = full_word
                self.shortcut_frequencies[shortcut] = 50
                added_manual += 1
        
        if added_manual > 0:
            print(f"✓ Added {added_manual} manual conversational shortcuts")
        
        # Show learned shortcuts
        if self.learned_shortcuts:
            print(f"\n📝 Total shortcuts: {len(self.learned_shortcuts)}")
            sorted_shortcuts = sorted(
                self.learned_shortcuts.items(),
                key=lambda x: self.shortcut_frequencies[x[0]],
                reverse=True
            )
            print("   Top 20 conversational shortcuts:")
            for shortcut, full_word in sorted_shortcuts[:20]:
                freq = self.shortcut_frequencies[shortcut]
                print(f"   '{shortcut}' → '{full_word}' (freq: {freq})")
    
    def load_shortcut_library(self):
        """Load additional shortcuts from GitHub CSV files"""
        # Try cache first
        if os.path.exists(SHORTCUT_LIBRARY_CACHE):
            try:
                with open(SHORTCUT_LIBRARY_CACHE, 'r', encoding='utf-8') as f:
                    self.csv_shortcuts = json.load(f)
                print(f"✓ Loaded {len(self.csv_shortcuts)} shortcuts from cache")
                return
            except Exception as e:
                print(f"⚠ Error loading cached shortcuts: {e}")
        
        # Download CSV files
        print("\n📚 Loading shortcut library from GitHub...")
        
        import urllib.request
        import csv
        import io
        
        for url in SHORTCUT_LIBRARY_URLS:
            try:
                print(f"  Downloading: {url.split('/')[-1]}...")
                
                response = urllib.request.urlopen(url, timeout=10)
                csv_data = response.read().decode('utf-8')
                
                csv_reader = csv.reader(io.StringIO(csv_data))
                next(csv_reader, None)  # Skip header
                
                count = 0
                for row in csv_reader:
                    if len(row) >= 2:
                        informal = row[0].strip().lower()
                        formal = row[1].strip().lower()
                        
                        if informal and formal and len(informal) < len(formal):
                            self.csv_shortcuts[informal] = formal
                            count += 1
                
                print(f"  ✓ Loaded {count} shortcuts from {url.split('/')[-1]}")
                
            except Exception as e:
                print(f"  ⚠ Error loading {url.split('/')[-1]}: {e}")
        
        # Save cache
        try:
            with open(SHORTCUT_LIBRARY_CACHE, 'w', encoding='utf-8') as f:
                json.dump(self.csv_shortcuts, f, indent=2, ensure_ascii=False)
            print(f"✓ Cached {len(self.csv_shortcuts)} shortcuts")
        except Exception as e:
            print(f"⚠ Error caching shortcuts: {e}")
    
    def resolve_shortcut(self, word):
        """Resolve a word to its full form if it's a shortcut"""
        word_lower = word.lower()
        
        # Priority: User > CSV > Dataset
        if word_lower in self.user_shortcuts:
            return self.user_shortcuts[word_lower]
        
        if word_lower in self.csv_shortcuts:
            return self.csv_shortcuts[word_lower]
        
        if word_lower in self.learned_shortcuts:
            return self.learned_shortcuts[word_lower]
        
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
        
        if shortcut_lower in self.learned_shortcuts:
            word = self.learned_shortcuts[shortcut_lower]
            if word not in [w for w, _ in expansions]:
                expansions.append((word, 'learned'))
        
        return expansions
    
    def learn_from_user_typing(self, typed_shortcut, selected_word):
        """Learn a shortcut pattern from user behavior"""
        typed_shortcut = typed_shortcut.lower()
        selected_word = selected_word.lower()
        
        if len(typed_shortcut) < len(selected_word) - 1:
            if typed_shortcut not in self.user_shortcuts:
                self.user_shortcuts[typed_shortcut] = selected_word
                print(f"🎓 Learned: '{typed_shortcut}' → '{selected_word}'")
            
            self.user_shortcut_usage[typed_shortcut] += 1
            self.save_user_learning()
    
    def add_new_word(self, word):
        """Add a new word to vocabulary"""
        word_lower = word.lower()
        if word_lower not in self.vocabulary:
            self.vocabulary.add(word_lower)
            self.new_words.add(word_lower)
            self.unigrams[word_lower] = 1
            print(f"📝 Added new word: '{word}'")
            self.save_user_learning()
    
    def track_word_usage(self, word, context=None):
        """Track user's word usage to improve predictions"""
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
        """Save user-specific learning data"""
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
            print(f"⚠ Error saving user learning: {e}")
    
    def load_user_learning(self):
        """Load user-specific learning data"""
        if not os.path.exists(USER_LEARNING_FILE):
            print("ℹ No user learning file found")
            return
        
        try:
            with open(USER_LEARNING_FILE, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            self.user_shortcuts = user_data.get('user_shortcuts', {})
            self.user_shortcut_usage = Counter(user_data.get('user_shortcut_usage', {}))
            self.new_words = set(user_data.get('new_words', []))
            self.word_usage_history = user_data.get('word_usage_history', [])
            
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
            'shortcut_frequencies': dict(self.shortcut_frequencies),
            'language_tags': self.language_tags
        }
        with open(NGRAM_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n✓ Saved conversational model to {NGRAM_CACHE_FILE}")
        print(f"✓ Vocabulary: {len(self.vocabulary)} words")
        print(f"✓ Shortcuts: {len(self.learned_shortcuts)}")
    
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
            self.learned_shortcuts = data.get('learned_shortcuts', {})
            self.shortcut_frequencies = Counter(data.get('shortcut_frequencies', {}))
            self.language_tags = data.get('language_tags', {})
            
            print(f"✓ Loaded cached conversational model")
            print(f"  Vocabulary: {len(self.vocabulary)} words")
            print(f"  Shortcuts: {len(self.learned_shortcuts)}")
            return True
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            return False
    
    def get_word_probability(self, word, context=None):
        """Calculate probability of word given context using n-gram model"""
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
        """Get word completions with conversational ranking"""
        prefix = prefix.lower()
        
        # Check for shortcut expansions
        shortcut_candidates = []
        all_expansions = self.get_all_shortcut_expansions(prefix)
        
        if all_expansions:
            for full_word, source in all_expansions:
                if source == 'user':
                    priority_multiplier = 3.0
                elif source == 'csv':
                    priority_multiplier = 2.0
                else:
                    priority_multiplier = 1.5
                
                shortcut_candidates.append((full_word, True, True, priority_multiplier))
        
        # Find exact prefix matches
        if len(prefix) == 1:
            min_word_length = 2
        else:
            min_word_length = max(len(prefix) + 1, 3)
        
        exact_matches = [
            word for word in self.vocabulary 
            if word.startswith(prefix) 
            and len(word) >= min_word_length
            and word.isalpha()
        ]
        
        # Get character-level completions
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        
        candidates = []
        for word in exact_matches:
            char_boost = 1.5 if word in char_completions else 1.0
            candidates.append((word, True, False, char_boost))
        
        # Score all candidates
        scored = []
        
        for word, is_exact, is_shortcut, boost in shortcut_candidates + candidates:
            prob = self.get_word_probability(word, context)
            
            exact_bonus = 1.5 if is_exact else 1.0
            shortcut_bonus = boost if is_shortcut else 1.0
            char_bonus = boost if not is_shortcut else 1.0
            
            final_score = prob * exact_bonus * shortcut_bonus * char_bonus
            
            scored.append((word, final_score))
        
        # Remove duplicates and sort
        seen = set()
        unique_scored = []
        for word, score in scored:
            if word not in seen:
                seen.add(word)
                unique_scored.append((word, score))
        
        unique_scored.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in unique_scored[:max_results]]
    
    def get_next_word_predictions(self, context, max_results=8):
        """Get next word predictions based on context"""
        if not context or len(context) == 0:
            most_common = self.unigrams.most_common(max_results)
            return [word for word, count in most_common]
        
        context_clean = [self.resolve_shortcut(w.lower()) for w in context]
        
        candidates = []
        
        # Try trigram
        if len(context_clean) >= 2:
            prev2, prev1 = context_clean[-2], context_clean[-1]
            trigram_key = (prev2, prev1)
            
            if trigram_key in self.trigrams:
                for word, count in self.trigrams[trigram_key].items():
                    prob = self.get_word_probability(word, context_clean)
                    candidates.append((word, prob, count))
        
        # Try bigram
        if len(context_clean) >= 1:
            prev = context_clean[-1]
            
            if prev in self.bigrams:
                for word, count in self.bigrams[prev].items():
                    if word not in [w for w, p, c in candidates]:
                        prob = self.get_word_probability(word, context_clean)
                        candidates.append((word, prob, count))
        
        # Add most common words if needed
        if len(candidates) < max_results:
            for word, count in self.unigrams.most_common(max_results * 2):
                if word not in [w for w, p, c in candidates]:
                    prob = self.get_word_probability(word, context_clean)
                    candidates.append((word, prob, count))
        
        # Sort by probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, prob, count in candidates[:max_results]]


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_current_token(text):
    """Extract the currently-being-typed word"""
    if not text:
        return None
    
    lines = text.split('\n')
    current_line = lines[-1]
    
    if current_line.endswith(' ') or current_line.endswith('\n'):
        return None
    
    words = current_line.split()
    if words:
        return words[-1].strip()
    return None


def get_context_words(text, n=2):
    """Get last n complete words before current token"""
    if not text:
        return []
    
    lines = text.split('\n')
    current_line = lines[-1]
    
    # Remove current incomplete word
    if current_line and not current_line.endswith(' '):
        words = current_line.split()[:-1]
    else:
        words = current_line.split()
    
    # Get previous lines if needed
    if len(words) < n and len(lines) > 1:
        for line in reversed(lines[:-1]):
            words = line.split() + words
            if len(words) >= n:
                break
    
    return words[-n:] if len(words) >= n else words


# ---------------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------------
class FilipinoIME(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Conversational Keyboard - Filipino & English")
        self.geometry("900x700")
        
        # Status
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main container
        main_container = ttk.Frame(self, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Text display
        text_frame = ttk.LabelFrame(main_container, text="Text Input", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_display = tk.Text(text_frame, wrap=tk.WORD, font=("Segoe UI", 12), height=8)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.text_display.bind("<KeyRelease>", lambda e: self.update_suggestions())
        self.text_display.bind("<Tab>", lambda e: self.auto_complete())
        
        # Suggestions area
        suggestions_frame = ttk.LabelFrame(main_container, text="Suggestions", padding="5")
        suggestions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Completions
        comp_label = ttk.Label(suggestions_frame, text="Complete current word:", font=("Segoe UI", 9, "bold"))
        comp_label.pack(anchor=tk.W)
        
        self.completion_container = ttk.Frame(suggestions_frame)
        self.completion_container.pack(fill=tk.X, pady=(2, 10))
        
        # Predictions
        pred_label = ttk.Label(suggestions_frame, text="Next word predictions:", font=("Segoe UI", 9, "bold"))
        pred_label.pack(anchor=tk.W)
        
        self.predictive_container = ttk.Frame(suggestions_frame)
        self.predictive_container.pack(fill=tk.X, pady=(2, 5))
        
        # Virtual keyboard
        keyboard_frame = ttk.LabelFrame(main_container, text="Virtual Keyboard", padding="5")
        keyboard_frame.pack(fill=tk.BOTH)
        
        self.create_keyboard(keyboard_frame)
        
        # Initialize suggestions
        self.update_suggestions()
    
    def update_suggestions(self, force_predictive=False):
        """Update completion and predictive suggestions"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        # Update completions
        for widget in self.completion_container.winfo_children():
            widget.destroy()
        
        if token and len(token) >= 1:
            completions = ngram_model.get_completion_suggestions(token, context, max_results=MAX_SUGGESTIONS)
            
            if completions:
                for word in completions:
                    # Check capitalization
                    should_capitalize = False
                    if token and token[0].isupper():
                        should_capitalize = True
                    elif not context:
                        should_capitalize = True
                    elif context and context[-1] in ['.', '!', '?']:
                        should_capitalize = True
                    elif context and context[-1][0].isupper():
                        should_capitalize = True
                    
                    display_word = word.capitalize() if should_capitalize else word
                    
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
                    text="No suggestions",
                    font=("Segoe UI", 9),
                    foreground="gray"
                )
                placeholder.pack()
        else:
            placeholder = ttk.Label(
                self.completion_container,
                text="Start typing for suggestions",
                font=("Segoe UI", 9, "italic"),
                foreground="gray"
            )
            placeholder.pack()
        
        # Update predictions
        for widget in self.predictive_container.winfo_children():
            widget.destroy()
        
        if not token or force_predictive:
            predictions = ngram_model.get_next_word_predictions(context, max_results=MAX_SUGGESTIONS)
            
            if predictions:
                for word in predictions:
                    should_capitalize = False
                    if not context:
                        should_capitalize = True
                    elif context and context[-1] in ['.', '!', '?']:
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
    
    def force_update_suggestions(self):
        """Force update with predictive suggestions"""
        self.update_suggestions(force_predictive=True)
    
    def apply_completion(self, word):
        """Apply word completion suggestion"""
        text = self.text_display.get("1.0", "end-1c")
        token = get_current_token(text)
        context = get_context_words(text, n=2)
        
        if token:
            # Learn from user
            ngram_model.learn_from_user_typing(token, word)
            
            # Replace token
            lines = text.split('\n')
            current_line = len(lines) - 1
            current_char = len(lines[-1]) - len(token)
            start_index = f"{current_line + 1}.{current_char}"
            
            self.text_display.delete(start_index, "end-1c")
            self.text_display.insert(start_index, word + " ")
        else:
            self.text_display.insert("end", word + " ")
        
        # Track usage
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
        
        # Track usage
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
                return "break"
    
    def insert_char(self, char):
        self.text_display.insert("insert", char)
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()
    
    def backspace(self):
        self.text_display.delete("insert-1c")
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()
    
    def space(self):
        self.text_display.insert("insert", " ")
        self.text_display.event_generate("<KeyRelease>")
        self.force_update_suggestions()
    
    def enter(self):
        self.text_display.insert("insert", "\n")
        self.text_display.event_generate("<KeyRelease>")
        self.update_suggestions()
    
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
# INITIALIZE AND RUN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONVERSATIONAL KEYBOARD - FILIPINO & ENGLISH")
    print("="*60)
    
    # Initialize n-gram model
    ngram_model = NgramModel()
    
    # Try to load cached model
    if not ngram_model.load_cache():
        # Train from conversational datasets
        ngram_model.train_from_conversational_datasets(['filipino', 'english'])
        
        # Load shortcut library
        ngram_model.load_shortcut_library()
        
        # Save cache
        ngram_model.save_cache()
    else:
        # Load shortcut library even if model is cached
        ngram_model.load_shortcut_library()
    
    # Load user learning
    ngram_model.load_user_learning()
    
    print("\n" + "="*60)
    print("MODEL READY - LAUNCHING GUI")
    print("="*60 + "\n")
    
    # Launch GUI
    app = FilipinoIME()
    app.mainloop()
