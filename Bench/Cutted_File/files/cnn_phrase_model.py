# =============================================================================
# cnn_phrase_model.py - Optional Text-CNN phrase suggestion ranker
# =============================================================================

import json
import os
import re
from collections import Counter

import config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PREDEFINED_FILE = os.path.join(_ROOT, "predefined_sentences.json")


AAC_SEED_PHRASES = [
    "i need help",
    "i need water",
    "i need food",
    "i am hungry",
    "i am thirsty",
    "i am tired",
    "i feel pain",
    "please help me",
    "please call my family",
    "please give me medicine",
    "can you help me",
    "can you move me",
    "i want to rest",
    "i want to sleep",
    "i want to go home",
    "thank you",
    "yes please",
    "no thank you",
    "kailangan ko ng tulong",
    "kailangan ko ng tubig",
    "kailangan ko ng pagkain",
    "gutom na ako",
    "uhaw na ako",
    "pagod na ako",
    "masakit ang katawan ko",
    "masakit ang ulo ko",
    "tulungan mo ako",
    "pakiusap tulungan mo ako",
    "pakitawagan ang pamilya ko",
    "gusto kong magpahinga",
    "gusto kong matulog",
    "salamat po",
    "oo po",
    "hindi po",
]


def _clean_token(token):
    token = str(token).lower().strip()
    token = re.sub(r"^[^\wñ']+|[^\wñ']+$", "", token, flags=re.IGNORECASE)
    return token


def _clean_sequence(seq):
    cleaned = []
    for token in seq:
        token = _clean_token(token)
        if token and (token.isalpha() or token.replace("'", "").isalpha()):
            cleaned.append(token)
    return cleaned


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class TextCnnPhraseRanker(nn.Module):
    def __init__(self, vocab_size, num_phrases, embed_dim=64, num_filters=32, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in (2, 3, 4)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.convs), num_phrases)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feature = F.relu(conv(emb))
            pooled.append(F.max_pool1d(feature, feature.size(2)).squeeze(2))
        z = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(z)


class CnnPhraseSuggester:
    def __init__(self):
        self.available = False
        self.disabled_reason = ""
        self.model = None
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.phrases = []
        self.phrase_tokens = []
        self.phrase_lang = []
        self.phrase_counts = Counter()
        self.max_context = getattr(config, "CNN_PHRASE_MAX_CONTEXT", 5)

    def load_or_train(self):
        if not getattr(config, "ENABLE_CNN_PHRASE_SUGGESTIONS", True):
            self.disabled_reason = "disabled in config"
            return False
        if torch is None:
            self.disabled_reason = "torch is not installed"
            return False
        if self.load_cache():
            return True
        return self.train_from_builtin()

    def _load_phrase_sources(self):
        phrase_map = {}

        def add_phrase(tokens, language, weight=1):
            tokens = _clean_sequence(tokens)
            if len(tokens) < 2:
                return
            phrase = " ".join(tokens)
            if phrase not in phrase_map:
                phrase_map[phrase] = {"tokens": tokens, "language": language, "count": 0}
            phrase_map[phrase]["count"] += weight
            if phrase_map[phrase]["language"] != language:
                phrase_map[phrase]["language"] = "both"

        for path, language in (
            (config.FILIPINO_DATASET_FILE, "filipino"),
            (config.ENGLISH_DATASET_FILE, "english"),
        ):
            data = _load_json(path) or {}
            for seq in data.get("corpus_sequences", []):
                add_phrase(seq, language, weight=3)
            for phrase in data.get("communication_corpus", []):
                add_phrase(str(phrase).split(), language, weight=2)

        for phrase in AAC_SEED_PHRASES:
            add_phrase(phrase.split(), "both", weight=8)

        saved = _load_json(_PREDEFINED_FILE) or {}
        if isinstance(saved, dict):
            for phrase, count in saved.items():
                try:
                    weight = max(1, int(count))
                except Exception:
                    weight = 1
                add_phrase(str(phrase).split(), "both", weight=weight + 4)

        rows = sorted(
            phrase_map.values(),
            key=lambda row: (-row["count"], row["language"], " ".join(row["tokens"])),
        )
        self.phrases = [" ".join(row["tokens"]) for row in rows]
        self.phrase_tokens = [row["tokens"] for row in rows]
        self.phrase_lang = [row["language"] for row in rows]
        self.phrase_counts = Counter({
            " ".join(row["tokens"]): int(row["count"])
            for row in rows
        })
        return len(self.phrases) >= 2

    def _build_training_examples(self):
        examples = []
        for phrase_id, tokens in enumerate(self.phrase_tokens):
            upper = min(len(tokens), self.max_context + 1)
            repeat = 1 + min(self.phrase_counts[self.phrases[phrase_id]] // 4, 4)
            for prefix_len in range(1, upper):
                prefix = tokens[:prefix_len]
                for _ in range(repeat):
                    examples.append((prefix, phrase_id))
        return examples

    def _ensure_vocab(self, examples):
        for tokens, _ in examples:
            for token in tokens:
                if token not in self.word_to_id:
                    self.word_to_id[token] = len(self.word_to_id)
        for tokens in self.phrase_tokens:
            for token in tokens:
                if token not in self.word_to_id:
                    self.word_to_id[token] = len(self.word_to_id)

    def _encode(self, tokens):
        tokens = _clean_sequence(tokens)[-self.max_context:]
        ids = [self.word_to_id.get(token, 1) for token in tokens]
        width = max(self.max_context, 4)
        while len(ids) < width:
            ids.insert(0, 0)
        return ids[-width:]

    def _tensorize(self, examples):
        encoded = [self._encode(tokens) for tokens, _ in examples]
        labels = [label for _, label in examples]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def train_from_builtin(self):
        if torch is None:
            self.disabled_reason = "torch is not installed"
            return False
        if not self._load_phrase_sources():
            self.disabled_reason = "not enough phrases"
            return False

        examples = self._build_training_examples()
        if not examples:
            self.disabled_reason = "no training examples"
            return False

        self._ensure_vocab(examples)
        x, y = self._tensorize(examples)
        self.model = TextCnnPhraseRanker(
            vocab_size=len(self.word_to_id),
            num_phrases=len(self.phrases),
        )
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        epochs = max(1, int(getattr(config, "CNN_PHRASE_EPOCHS", 12)))
        batch_size = 64
        for _ in range(epochs):
            order = torch.randperm(x.size(0))
            for start in range(0, x.size(0), batch_size):
                idx = order[start:start + batch_size]
                logits = self.model(x[idx])
                loss = F.cross_entropy(logits, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.available = True
        self.save_cache()
        print(f"[OK] CNN phrase model trained - phrases: {len(self.phrases)}, examples: {len(examples)}")
        return True

    def save_cache(self):
        if torch is None or self.model is None:
            return
        payload = {
            "word_to_id": self.word_to_id,
            "phrases": self.phrases,
            "phrase_tokens": self.phrase_tokens,
            "phrase_lang": self.phrase_lang,
            "phrase_counts": dict(self.phrase_counts),
            "max_context": self.max_context,
            "state_dict": self.model.state_dict(),
        }
        try:
            torch.save(payload, config.CNN_PHRASE_CACHE_FILE)
        except Exception as exc:
            print(f"[Warn] Error saving CNN phrase cache: {exc}")

    def load_cache(self):
        if torch is None or not os.path.exists(config.CNN_PHRASE_CACHE_FILE):
            return False
        try:
            payload = torch.load(config.CNN_PHRASE_CACHE_FILE, map_location="cpu")
            self.word_to_id = payload["word_to_id"]
            self.phrases = payload["phrases"]
            self.phrase_tokens = payload["phrase_tokens"]
            self.phrase_lang = payload.get("phrase_lang", ["both"] * len(self.phrases))
            self.phrase_counts = Counter(payload.get("phrase_counts", {}))
            self.max_context = payload.get("max_context", self.max_context)
            self.model = TextCnnPhraseRanker(
                vocab_size=len(self.word_to_id),
                num_phrases=len(self.phrases),
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
            self.available = True
            print(f"[OK] CNN phrase cache loaded - phrases: {len(self.phrases)}")
            return True
        except Exception as exc:
            self.disabled_reason = f"cache load failed: {exc}"
            return False

    def _language_allowed(self, phrase_id, language):
        if language == "both":
            return True
        phrase_lang = self.phrase_lang[phrase_id] if phrase_id < len(self.phrase_lang) else "both"
        return phrase_lang in (language, "both")

    def _prefix_bonus(self, context_tokens, phrase_tokens):
        if not context_tokens:
            return 0.0
        if phrase_tokens[:len(context_tokens)] == context_tokens:
            return 2.0 + 0.3 * len(context_tokens)
        joined_context = " ".join(context_tokens)
        joined_phrase = " ".join(phrase_tokens)
        if joined_phrase.startswith(joined_context):
            return 1.0
        return -1.0

    def get_phrase_suggestions(self, context, max_results=2, language="both"):
        if not self.available or self.model is None or torch is None:
            return []
        context_tokens = _clean_sequence(context or [])[-self.max_context:]
        if not context_tokens:
            return []

        encoded = torch.tensor([self._encode(context_tokens)], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(encoded), dim=1)[0]

        scored = []
        for phrase_id, phrase in enumerate(self.phrases):
            if not self._language_allowed(phrase_id, language):
                continue
            tokens = self.phrase_tokens[phrase_id]
            if len(tokens) <= len(context_tokens):
                continue
            if tokens[:len(context_tokens)] == context_tokens:
                missing = tokens[len(context_tokens):]
            else:
                missing = tokens
            if not missing:
                continue
            count_bonus = min(self.phrase_counts.get(phrase, 0), 20) / 20.0
            score = float(probs[phrase_id]) + self._prefix_bonus(context_tokens, tokens) + count_bonus
            if score <= 0:
                continue
            scored.append((phrase, missing, score))

        scored.sort(key=lambda item: item[2], reverse=True)
        results = []
        seen = set()
        for phrase, missing, _ in scored:
            if phrase in seen:
                continue
            seen.add(phrase)
            results.append({
                "phrase": phrase,
                "missing_words": missing,
            })
            if len(results) >= max_results:
                break
        return results


cnn_phrase_model = CnnPhraseSuggester()
