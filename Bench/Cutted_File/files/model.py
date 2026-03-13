# =============================================================================
# model.py — N-gram language model, typo tolerance, and helper utilities
# =============================================================================

import os
import json
import pickle
from collections import defaultdict, Counter

from config import NGRAM_CACHE_FILE, USER_LEARNING_FILE, MODEL_VERSION
from dataset import FILIPINO_WORDS, FILIPINO_SHORTCUTS, COMMUNICATION_CORPUS


# =============================================================================
# DAMERAU-LEVENSHTEIN DISTANCE  (Typo Tolerance)
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
# N-GRAM LANGUAGE MODEL
# =============================================================================
class NgramModel:
    def __init__(self):
        self.unigrams           = Counter()
        self.bigrams            = defaultdict(Counter)
        self.trigrams           = defaultdict(Counter)
        self.vocabulary         = set()
        self.total_words        = 0
        self.char_bigrams       = defaultdict(Counter)
        self.char_trigrams      = defaultdict(Counter)
        self.csv_shortcuts      = {}
        self.user_shortcuts     = {}
        self.user_shortcut_usage = Counter()
        self.new_words          = set()
        self.word_usage_history = []

    # ── Training ──────────────────────────────────────────────────────────────
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
            "nlng": "nalang", "lng": "lang",   "nmn": "naman",
            "ksi":  "kasi",   "kse": "kasi",   "kng": "kung",   "khit": "kahit",
            "d2":   "dito",   "dn":  "doon",   "dun": "doon",
            "pde":  "pwede",  "pwd": "pwede",
            "sna":  "sana",   "tlg": "talaga", "tlga": "talaga",
            "sya":  "siya",   "xa":  "siya",   "aq":  "ako",    "q":    "ako",
            "nde":  "hindi",  "hnd": "hindi",  "di":  "hindi",
            "pra":  "para",   "wla": "wala",   "my":  "may",
            "ung":  "yung",   "un":  "yun",    "dba": "diba",   "db":   "diba",
            "bat":  "bakit",  "bkt": "bakit",  "pno": "paano",  "pano": "paano",
            "tyo":  "tayo",   "kyo": "kayo",   "kmi": "kami",   "cla":  "sila",
            "ikw":  "ikaw",   "ako": "ako"
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
                self.bigrams[all_tokens[i-1]][token] += 1
            if i > 1:
                self.trigrams[(all_tokens[i-2], all_tokens[i-1])][token] += 1
        print(f"✓ Vocabulary: {len(self.vocabulary)} unique words")

    # ── Character n-grams ─────────────────────────────────────────────────────
    def _has_vowels(self, word):
        return any(c in 'aeiouAEIOU' for c in word)

    def _build_char_ngrams(self, word):
        if len(word) < 2:
            return
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            if i < len(word) - 2:
                self.char_bigrams[bigram][word[i+2]] += 1
            else:
                self.char_bigrams[bigram]["<END>"] += 1
        for i in range(len(word) - 2):
            trigram = (word[i], word[i+1])
            if i < len(word) - 3:
                self.char_trigrams[trigram][word[i+3]] += 1

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
            scored.append((word, score + freq_score * 10))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored[:max_results]]

    # ── Shortcuts ─────────────────────────────────────────────────────────────
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

    # ── User learning ─────────────────────────────────────────────────────────
    def learn_from_user_typing(self, typed_shortcut, selected_word):
        typed_shortcut = typed_shortcut.lower()
        selected_word  = selected_word.lower()
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
            prev = self.resolve_shortcut(context[-1].lower())
            self.bigrams[prev][word_lower] += 1
        if context and len(context) >= 2:
            prev2 = self.resolve_shortcut(context[-2].lower())
            prev1 = self.resolve_shortcut(context[-1].lower())
            self.trigrams[(prev2, prev1)][word_lower] += 1
        self.word_usage_history.append((word_lower, context))
        if len(self.word_usage_history) > 1000:
            self.word_usage_history.pop(0)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_user_learning(self):
        user_data = {
            'user_shortcuts':      self.user_shortcuts,
            'user_shortcut_usage': dict(self.user_shortcut_usage),
            'new_words':           list(self.new_words),
            'word_usage_history':  self.word_usage_history[-100:]
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
            self.user_shortcuts      = user_data.get('user_shortcuts', {})
            self.user_shortcut_usage = Counter(user_data.get('user_shortcut_usage', {}))
            self.new_words           = set(user_data.get('new_words', []))
            self.word_usage_history  = user_data.get('word_usage_history', [])
            self.vocabulary.update(self.new_words)
            print(f"✓ User shortcuts: {len(self.user_shortcuts)}")
            print(f"✓ New words: {len(self.new_words)}")
        except Exception as e:
            print(f"⚠ Error loading: {e}")

    def save_cache(self):
        data = {
            'version':      MODEL_VERSION,
            'unigrams':     dict(self.unigrams),
            'bigrams':      {k: dict(v) for k, v in self.bigrams.items()},
            'trigrams':     {k: dict(v) for k, v in self.trigrams.items()},
            'char_bigrams': {k: dict(v) for k, v in self.char_bigrams.items()},
            'char_trigrams':{k: dict(v) for k, v in self.char_trigrams.items()},
            'vocabulary':   list(self.vocabulary),
            'total_words':  self.total_words,
            'csv_shortcuts':self.csv_shortcuts,
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
            if data.get('version', '1.0') != MODEL_VERSION:
                print("⚠ Cache version mismatch, rebuilding...")
                return False
            self.unigrams    = Counter(data['unigrams'])
            self.bigrams     = defaultdict(Counter)
            self.trigrams    = defaultdict(Counter)
            self.char_bigrams  = defaultdict(Counter)
            self.char_trigrams = defaultdict(Counter)
            for k, v in data['bigrams'].items():
                self.bigrams[k] = Counter(v)
            for k, v in data['trigrams'].items():
                self.trigrams[k] = Counter(v)
            for k, v in data.get('char_bigrams', {}).items():
                self.char_bigrams[k] = Counter(v)
            for k, v in data.get('char_trigrams', {}).items():
                self.char_trigrams[k] = Counter(v)
            self.vocabulary  = set(data['vocabulary'])
            self.total_words = data['total_words']
            self.csv_shortcuts = data.get('csv_shortcuts', {})
            print(f"✓ Loaded cache — Vocabulary: {len(self.vocabulary)}, Shortcuts: {len(self.csv_shortcuts)}")
            return True
        except Exception as e:
            print(f"⚠ Error: {e}")
            return False

    # ── Probability & suggestions ─────────────────────────────────────────────
    def get_word_probability(self, word, context=None):
        word  = word.lower()
        alpha = 0.1
        vocab_size = len(self.vocabulary)
        if not context:
            count = self.unigrams.get(word, 0)
            return (count + alpha) / (self.total_words + alpha * vocab_size)
        elif len(context) == 1:
            prev = self.resolve_shortcut(context[0].lower())
            count = self.bigrams[prev].get(word, 0)
            prev_count = self.unigrams.get(prev, 0)
            if prev_count == 0:
                return self.get_word_probability(word)
            return (count + alpha) / (prev_count + alpha * vocab_size)
        else:
            prev2 = self.resolve_shortcut(context[-2].lower())
            prev1 = self.resolve_shortcut(context[-1].lower())
            ctx   = (prev2, prev1)
            count = self.trigrams[ctx].get(word, 0)
            ctx_count = sum(self.trigrams[ctx].values())
            if ctx_count == 0:
                return self.get_word_probability(word, [prev1])
            return (count + alpha) / (ctx_count + alpha * vocab_size)

    def get_completion_suggestions(self, prefix, context=None, max_results=8):
        prefix = prefix.lower()
        shortcut_candidates = []
        for full_word, source in self.get_all_shortcut_expansions(prefix):
            mult = 10.0 if source == 'user' else 8.0
            shortcut_candidates.append((full_word, False, True, mult))

        min_word_length = 2
        exact_matches = [
            w for w in self.vocabulary
            if w.startswith(prefix) and len(w) >= min_word_length
            and w.isalpha() and self._has_vowels(w)
        ]
        char_completions = self.get_char_level_completions(prefix, max_results=10)
        candidates = [
            (w, True, False, 2.0 if w in char_completions else 1.0)
            for w in exact_matches
        ]

        fuzzy_candidates = []
        if len(prefix) >= 2:
            for word in self.vocabulary:
                if word in exact_matches or len(word) < min_word_length:
                    continue
                if not word.isalpha() or not self._has_vowels(word):
                    continue
                if prefix in word:
                    pos = word.index(prefix)
                    fuzzy_candidates.append((word, False, False, 0.5 / (pos + 1)))
                elif abs(len(word) - len(prefix)) <= 2:
                    dist = damerau_levenshtein_distance(prefix, word[:len(prefix)])
                    if dist == 1:
                        fuzzy_candidates.append((word, False, False, (1.0 / (1.0 + dist)) * 0.2))

        scored = []
        for word, is_exact, is_shortcut, mult in shortcut_candidates + candidates + fuzzy_candidates:
            prob = self.get_word_probability(word, context)
            if is_shortcut:
                score = prob * mult * 1000
                if prefix in self.user_shortcut_usage:
                    score *= (1.0 + min(self.user_shortcut_usage[prefix] / 10.0, 2.0))
            elif is_exact:
                score = prob * mult * 10
            else:
                score = prob * mult
            scored.append((word, score))

        seen = {}
        for word, score in scored:
            if word not in seen or score > seen[word]:
                seen[word] = score
        unique = sorted(seen.items(), key=lambda x: -x[1])
        return [w for w, _ in unique[:max_results]]

    def get_next_word_suggestions(self, context=None, max_results=6):
        if not context:
            return [w for w, _ in self.unigrams.most_common(max_results)]
        elif len(context) == 1:
            prev = self.resolve_shortcut(context[0].lower())
            source = self.bigrams[prev] if prev in self.bigrams else self.unigrams
            return [w for w, _ in source.most_common(max_results)]
        else:
            prev2 = self.resolve_shortcut(context[-2].lower())
            prev1 = self.resolve_shortcut(context[-1].lower())
            ctx   = (prev2, prev1)
            if ctx in self.trigrams:
                return [w for w, _ in self.trigrams[ctx].most_common(max_results)]
            elif prev1 in self.bigrams:
                return [w for w, _ in self.bigrams[prev1].most_common(max_results)]
            else:
                return [w for w, _ in self.unigrams.most_common(max_results)]


# =============================================================================
# Shared model instance — imported by ui.py and main.py
# =============================================================================
ngram_model = NgramModel()
