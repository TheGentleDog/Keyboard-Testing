"""
generate_dataset_english.py — English Dataset Generator using RoBERTa-base

Generates english_dataset.json from scratch using:
  roberta-base (masked language model)

Mirrors generate_dataset.py exactly — same pipeline, same output schema,
including the `corpus_sequences` field used by model.py for n-gram training.

Can be run standalone:
    python generate_dataset_english.py

Or called from main.py on first startup via generate_if_missing().
"""

from collections import defaultdict, Counter
import json
import re
from datetime import date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME  = "roberta-base"
TOP_K       = 50
MIN_SCORE   = 0.003
OUTPUT_FILE = "english_dataset.json"

# ─────────────────────────────────────────────
# SEED TEMPLATES
# ─────────────────────────────────────────────
TEMPLATES = {
    "pronouns": [
        "<mask> is hungry now.",
        "<mask> is sleeping.",
        "Has <mask> eaten yet?",
        "Where is <mask>?",
        "<mask> is happy today.",
        "<mask> is going home.",
        "I am with <mask>.",
        "This is for <mask>.",
        "<mask> said to come already.",
        "I am looking for <mask>.",
        "<mask> is very talented.",
        "<mask> is already here.",
    ],
    "verbs": [
        "I want to <mask> some food.",
        "She needs to <mask> some water.",
        "He is <mask> in the room.",
        "Can I <mask> here?",
        "Can you <mask> later?",
        "I don't want to <mask> right now.",
        "She wants to <mask> early.",
        "She started to <mask>.",
        "He keeps <mask> every day.",
        "We need to <mask> quickly.",
        "Go ahead and <mask>.",
        "Help me <mask>.",
        "I don't feel like <mask> today.",
        "Let's just <mask> already.",
        "It's time to <mask>.",
        "You can <mask> now.",
        "Let's try to <mask> tomorrow.",
        "No need to <mask> anymore.",
        "Can you <mask> right now?",
        "Who is <mask> there?",
        "Why are you <mask> here?",
        "What time are you <mask>?",
    ],
    "adjectives": [
        "The food is really <mask>.",
        "The child was <mask> yesterday.",
        "I feel <mask> today.",
        "Your work is already <mask>.",
        "The place is <mask> and quiet.",
        "The weather is so <mask> today.",
        "Her face looks <mask>.",
        "The water is <mask> now.",
        "He felt <mask> in his body.",
        "Her voice is <mask>.",
        "Her clothes are <mask>.",
        "Their food was <mask>.",
        "The room is <mask> and clean.",
        "His job is <mask>.",
        "Her attitude is <mask>.",
        "The price of this is <mask>.",
        "The road is <mask> and muddy.",
    ],
    "nouns_people": [
        "My <mask> is here.",
        "The <mask> went to the hospital.",
        "My <mask> said to eat already.",
        "I am looking for my <mask>.",
        "My <mask> is very good at it.",
        "I am with my <mask>.",
        "The <mask> is working at the office.",
        "The <mask> called mom.",
        "Our <mask> is very careful.",
        "The <mask> has arrived.",
        "The <mask> asked the teacher.",
        "The <mask> agreed to the request.",
        "The <mask> was the one who took care of him.",
    ],
    "nouns_places": [
        "She went to the <mask>.",
        "He is over at the <mask>.",
        "We need to go to the <mask>.",
        "The <mask> is just nearby.",
        "We are waiting at the <mask>.",
        "I will go out to the <mask> later.",
        "The <mask> is far from here.",
        "They built a <mask> nearby.",
        "The <mask> is beautiful in the afternoon.",
        "There are many people at the <mask> now.",
        "She came from the <mask>.",
        "Food is expensive at the <mask>.",
        "We are almost at the <mask>.",
    ],
    "nouns_things": [
        "I need a <mask>.",
        "Where is my <mask>?",
        "Give me the <mask>.",
        "He is using a <mask>.",
        "I lost my <mask>.",
        "His <mask> is brand new.",
        "Our <mask> is already broken.",
        "She bought a <mask> yesterday.",
        "The <mask> is expensive now.",
        "Can I borrow the <mask>?",
        "Put the <mask> over there.",
        "I brought my <mask>.",
        "We have no more <mask>.",
        "We need to fix the <mask>.",
    ],
    "food_drink": [
        "I want to eat some <mask>.",
        "The <mask> here is delicious.",
        "He ordered some <mask>.",
        "She wants to drink some <mask>.",
        "My favorite dish is <mask>.",
        "She cooked <mask> for dinner.",
        "There is no <mask> in the fridge.",
        "We bought some <mask> at the store.",
        "I'm hungry, I want some <mask>.",
        "The <mask> is nutritious for the body.",
        "Our breakfast was <mask>.",
        "Prepare the <mask> for the guests.",
        "What kind of <mask> do you want?",
        "The kids love <mask>.",
        "The <mask> is still hot.",
        "I'm thirsty, can you pass the <mask>?",
    ],
    "time": [
        "We will go <mask>.",
        "We will do this <mask>.",
        "She started back <mask>.",
        "I will wait for you <mask>.",
        "He came back <mask>.",
        "We will meet <mask>.",
        "It was still early <mask>.",
        "The <mask> is already over.",
        "We have been waiting since <mask>.",
        "Since <mask> he has already changed.",
        "We only have until <mask>.",
        "What are your plans for <mask>?",
    ],
    "particles": [
        "Okay <mask>, let's go.",
        "I want to <mask> eat.",
        "No <mask>, I won't go.",
        "Can we <mask>?",
        "We need to <mask> go home.",
        "You are <mask> right.",
        "That's just how <mask> she is.",
        "I love you <mask>.",
        "Everything is <mask> okay.",
        "Come on <mask>, let's leave.",
        "Don't be <mask> sad.",
        "Let's go <mask> together.",
        "Let's <mask> go tomorrow.",
        "Let's <mask> eat first.",
    ],
    "questions": [
        "<mask> is your name?",
        "<mask> did she go?",
        "<mask> happened?",
        "<mask> did you do this?",
        "<mask> are we going home?",
        "<mask> do you want?",
        "<mask> is the problem?",
        "<mask> is she now?",
        "<mask> did you go there?",
        "<mask> is next?",
        "<mask> is beautiful here?",
        "<mask> do you need?",
    ],
    "expressions": [
        "<mask>, thank you for your help.",
        "<mask>, I don't know.",
        "Just <mask> then.",
        "<mask>, that's okay.",
        "Let's <mask> already.",
        "My answer is <mask>.",
        "I already said <mask>.",
        "<mask>, that's really how it is.",
        "I can't <mask> that.",
        "I know already, <mask>.",
        "What is <mask>.",
        "That's just <mask>.",
    ],
    "location": [
        "I'm just here <mask>.",
        "She moved <mask>.",
        "We went <mask> the house.",
        "He is waiting <mask>.",
        "Find it <mask>.",
        "Put it <mask>.",
        "Everything is over <mask>.",
        "From <mask> all the way there.",
        "Go <mask>.",
        "Everything is at the <mask>.",
    ],
    "body": [
        "My <mask> is hurting.",
        "She touched my <mask>.",
        "He fixed his <mask>.",
        "My <mask> is hot.",
        "My <mask> felt the pain.",
        "My <mask> is tired.",
        "His <mask> is very healthy.",
        "His <mask> was injured.",
        "He squeezed his <mask>.",
        "His <mask> is strong.",
    ],
    "emotions": [
        "She felt <mask> in her heart.",
        "He was so <mask> inside.",
        "She couldn't stop being <mask>.",
        "Her <mask> was obvious.",
        "She felt like she was <mask>.",
        "She became <mask> when she heard that.",
        "Her <mask> was visible on her face.",
        "He hid because of <mask>.",
        "He couldn't accept the <mask>.",
        "Her heart was full of <mask>.",
        "The <mask> was not easy to bear.",
    ],
    "slang": [
        "Wow, she is really <mask>.",
        "Something super <mask> happened.",
        "That is literally <mask>.",
        "For me that is just <mask>.",
        "I feel like it's already <mask>.",
        "I'm really into your <mask>.",
        "I wish everyone was <mask>.",
        "That is so <mask>.",
        "Just kidding, it's just <mask>.",
        "That's so <mask> of you.",
        "That is totally <mask>.",
        "Is that really <mask>?",
        "What is your <mask>?",
        "Don't be <mask> here.",
        "You're always like that, always <mask>.",
        "You did it again, always <mask>.",
    ],
    "health_needs": [
        "I need some <mask> right now.",
        "My <mask> is hurting right now.",
        "I need a <mask>.",
        "Please give me some <mask>.",
        "Can I get some <mask>?",
        "There is no <mask> here.",
        "I need help with my <mask>.",
        "My <mask> is not doing well.",
        "Let's go to the <mask>.",
        "She asked the doctor for <mask>.",
        "He takes his <mask> every day.",
        "He needs <mask> to get better.",
        "She got a <mask> yesterday.",
        "The <mask> is a remedy for illness.",
    ],
    "communication": [
        "Tell her that <mask>.",
        "Please call <mask>.",
        "I want to talk to <mask>.",
        "Explain to me the <mask>.",
        "Please text me the <mask>.",
        "Please send me the <mask>.",
        "She said that <mask>.",
        "She asked if <mask>.",
        "She passed along the <mask>.",
        "He messaged about the <mask>.",
        "Tell him about the <mask>.",
        "Please tell <mask> to come already.",
    ],
    "school_work": [
        "We have a <mask> tomorrow.",
        "We need to finish the <mask>.",
        "Our <mask> is really difficult.",
        "She is studying <mask>.",
        "Our teacher in <mask> is very strict.",
        "My <mask> is already submitted.",
        "I still have a <mask> to do.",
        "My <mask> deadline is tomorrow.",
        "He had a hard time with <mask>.",
        "She is really good at <mask>.",
        "Our <mask> is scheduled early.",
        "The teacher gave us a <mask>.",
    ],
    "daily_life": [
        "Every morning, she first <mask>.",
        "Before sleeping, she always <mask>.",
        "After eating, she usually <mask>.",
        "While waiting, she <mask>.",
        "At home, she often <mask>.",
        "Every night, they <mask>.",
        "Upon arriving at work, she immediately <mask>.",
        "While playing, the child <mask>.",
        "During vacation, we always <mask>.",
        "On the road, he <mask>.",
    ],
}

CATEGORY_MAP = {
    "emotions":     "expressions",
    "slang":        "slang",
    "health_needs": "expressions",
    "communication":"expressions",
    "school_work":  "nouns_things",
    "daily_life":   "verbs",
}

SHORTCUTS = {
    "u": "you", "ur": "your", "r": "are", "b": "be",
    "y": "why", "bc": "because", "bcz": "because",
    "cuz": "because", "cos": "because",
    "tho": "though", "thru": "through",
    "tmr": "tomorrow", "tmrw": "tomorrow",
    "rn": "right now", "atm": "at the moment",
    "asap": "as soon as possible",
    "imo": "in my opinion", "imho": "in my honest opinion",
    "tbh": "to be honest", "ngl": "not gonna lie",
    "fyi": "for your information", "btw": "by the way",
    "idk": "I don't know", "ik": "I know", "ikr": "I know right",
    "nvm": "never mind", "nm": "never mind",
    "omg": "oh my god", "wtf": "what the heck", "wth": "what the heck",
    "lol": "laughing out loud", "lmao": "laughing so hard",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head", "irl": "in real life",
    "brb": "be right back", "afk": "away from keyboard",
    "g2g": "got to go", "gtg": "got to go",
    "ttyl": "talk to you later", "ttys": "talk to you soon",
    "hmu": "hit me up", "dm": "direct message", "pm": "private message",
    "np": "no problem", "ty": "thank you",
    "thx": "thanks", "thnx": "thanks",
    "pls": "please", "plz": "please",
    "ok": "okay", "k": "okay",
    "gr8": "great", "l8r": "later",
    "tbf": "to be fair", "fr": "for real", "frfr": "for real for real",
    "hbd": "happy birthday", "hny": "happy new year",
    "gm": "good morning", "gn": "good night",
    "ga": "good afternoon", "ge": "good evening",
    "tc": "take care", "imy": "I miss you",
    "ily": "I love you", "ilu": "I love you",
    "bff": "best friend forever",
    "lmk": "let me know",
    "wyd": "what are you doing", "wbu": "what about you",
    "gg": "good game", "wp": "well played",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_valid_token(token: str) -> bool:
    token = token.strip()
    if len(token) < 2:
        return False
    # allow apostrophes for contractions (don't, it's, I'm …)
    if not re.match(r"^[a-zA-Z']+$", token):
        return False
    return True


def expand_verb_forms(word: str) -> list:
    """Generate common English verb inflections."""
    variants = [word]
    if word.endswith("e") and len(word) > 2:
        variants.append(word[:-1] + "ing")
        variants.append(word + "d")
    else:
        variants.append(word + "ing")
        variants.append(word + "ed")
    if word.endswith(("s", "sh", "ch", "x", "z")):
        variants.append(word + "es")
    else:
        variants.append(word + "s")
    return list(set(variants))


def top_words(counter, n=100):
    return [w for w, _ in counter.most_common(n)]


# ─────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────
def generate(output_file: str = OUTPUT_FILE):
    """
    Run the full generation pipeline and save to output_file.
    Called by main.py on first startup, or run standalone.
    """
    from transformers import pipeline as hf_pipeline

    print(f"🤖 Loading {MODEL_NAME} ...")
    fill = hf_pipeline("fill-mask", model=MODEL_NAME, top_k=TOP_K)
    print("✓ Model loaded.\n")

    print("🔍 Generating vocabulary from masked sentences...\n")
    category_words   = defaultdict(Counter)
    all_corpus_seeds = []
    total_templates  = sum(len(t) for t in TEMPLATES.values())
    processed        = 0

    for category, templates in TEMPLATES.items():
        print(f"  ▸ {category} ({len(templates)} templates)")
        for tmpl in templates:
            processed += 1
            progress = int((processed / total_templates) * 40)
            bar = "█" * progress + "░" * (40 - progress)
            print(f"\r    [{bar}] {processed}/{total_templates}", end="", flush=True)
            try:
                results = fill(tmpl)
                for r in results:
                    token = r["token_str"].strip().lower()
                    score = r["score"]
                    if score < MIN_SCORE or not is_valid_token(token):
                        continue
                    category_words[category][token] += score
                    phrase = re.sub(r'\s+', ' ', tmpl.replace("<mask>", token)).strip().rstrip('.')
                    all_corpus_seeds.append(phrase.lower())
            except Exception:
                pass

    print(f"\r    [{'█'*40}] {total_templates}/{total_templates} ✓\n")

    print("🔧 Expanding verb forms...")
    for base in list(category_words.get("verbs", Counter()).keys())[:40]:
        for variant in expand_verb_forms(base):
            if is_valid_token(variant) and variant not in category_words["verbs"]:
                category_words["verbs"][variant] += 0.001
    print("✓ Done.\n")

    # ── Post-process vocabulary ───────────────────────────────────────────────
    vocabulary     = {}
    all_words_flat = set()
    for cat, counter in category_words.items():
        mapped = CATEGORY_MAP.get(cat, cat)
        if mapped not in vocabulary:
            vocabulary[mapped] = []
        for w in top_words(counter, n=100):
            if w not in all_words_flat:
                vocabulary[mapped].append(w)
                all_words_flat.add(w)

    # ── communication_corpus: deduplicated phrase strings (legacy / display)
    seen_phrases = set()
    corpus = []
    for phrase in all_corpus_seeds:
        clean = phrase.strip()
        if clean not in seen_phrases and len(clean.split()) >= 2:
            corpus.append(clean)
            seen_phrases.add(clean)
        if len(corpus) >= 800:
            break

    # ── corpus_sequences: tokenised lists — used by model.py for n-gram training
    # Each entry is a list of word strings preserving sequential order so that
    # bigram/trigram co-occurrence counts reflect real phrase structure.
    corpus_sequences = []
    seen_seq = set()
    for phrase in all_corpus_seeds:
        tokens = phrase.strip().lower().split()
        if len(tokens) >= 2:
            key = " ".join(tokens)
            if key not in seen_seq:
                corpus_sequences.append(tokens)
                seen_seq.add(key)
        if len(corpus_sequences) >= 1200:
            break

    total_words     = sum(len(v) for v in vocabulary.values())
    total_phrases   = len(corpus)
    total_shortcuts = len(SHORTCUTS)

    print(f"📊 Results:")
    print(f"   Vocabulary categories : {len(vocabulary)}")
    print(f"   Total unique words    : {total_words}")
    print(f"   Corpus phrases        : {total_phrases}")
    print(f"   Corpus sequences      : {len(corpus_sequences)}")
    print(f"   Shortcuts             : {total_shortcuts}")
    for cat, words in vocabulary.items():
        print(f"     {cat:<22} {len(words)} words")

    dataset = {
        "metadata": {
            "version":          "1.0-roberta",
            "language":         "English (RoBERTa-generated)",
            "description":      "Auto-generated Dataset via roberta-base — For Gaze Based Digital Keyboard",
            "total_words":      total_words,
            "total_phrases":    total_phrases,
            "total_shortcuts":  total_shortcuts,
            "last_updated":     str(date.today()),
            "model_used":       MODEL_NAME,
        },
        "vocabulary":           vocabulary,
        "shortcuts":            SHORTCUTS,
        "communication_corpus": corpus,
        # Sequential token lists for n-gram training in model.py
        "corpus_sequences":     corpus_sequences,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved → {output_file}\n")


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
def generate_if_missing(output_file: str = OUTPUT_FILE):
    """Called by main.py — only regenerates if the file is absent."""
    import os
    if not os.path.exists(output_file):
        print(f"⚠  {output_file} not found — generating now...")
        generate(output_file)
    else:
        print(f"✓ Dataset found: {output_file}")


if __name__ == "__main__":
    generate()
