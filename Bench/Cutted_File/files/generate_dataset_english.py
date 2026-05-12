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
import os
import re
from datetime import date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME  = "roberta-base"
TOP_K       = 50
MIN_SCORE   = 0.003
_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(_HERE, "english_dataset.json")

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

# ─────────────────────────────────────────────
# FILTER SETS — applied after vocabulary generation
# ─────────────────────────────────────────────

# Any proper noun — people (politicians, religious, celebrities, common first
# names), places (countries, cities, continents), and religions/deities.
# The model frequently predicts these in person/place slots and they should
# not appear as general vocabulary words.
ENGLISH_PROPER_NAMES = {
    # ── US politicians & presidents ──────────────────────────────────────
    "trump", "biden", "obama", "clinton", "bush", "reagan", "carter",
    "lincoln", "washington", "jefferson", "kennedy", "nixon", "johnson",
    "harris", "pence", "cheney", "pelosi", "mcconnell", "schumer",
    "sanders", "warren", "aoc", "desantis", "newsom", "abbott",
    "roosevelt", "eisenhower", "truman", "wilson", "adams", "monroe",
    "hillary",
    # ── UK politicians ───────────────────────────────────────────────────
    "sunak", "starmer", "thatcher", "blair", "cameron", "brown",
    "major", "heath", "callaghan", "attlee", "churchill",
    # ── International politicians ────────────────────────────────────────
    "putin", "zelensky", "xi", "jinping", "modi", "trudeau", "macron",
    "scholz", "meloni", "kim", "netanyahu", "erdogan", "bolsonaro",
    "lula", "milei", "orban", "lukashenko", "marcos", "duterte",
    "mandela", "gandhi", "hitler", "mussolini", "stalin", "mao",
    "castro", "chavez", "pinochet", "franco",
    # ── Religious figures & deities ──────────────────────────────────────
    "jesus", "christ", "god", "allah", "buddha", "muhammad", "moses",
    "abraham", "noah", "adam", "eve", "satan", "lucifer", "devil",
    "mary", "joseph", "peter", "paul", "john", "matthew", "mark",
    "luke", "james", "thomas", "andrew", "philip", "judas", "david",
    "solomon", "elijah", "elias", "gabriel", "michael", "raphael",
    "vishnu", "shiva", "krishna", "rama", "brahma", "zeus", "apollo",
    "thor", "odin", "hercules", "poseidon", "athena", "hera",
    # ── Common Western first names ───────────────────────────────────────
    "james", "john", "robert", "michael", "william", "richard", "charles",
    "joseph", "thomas", "christopher", "daniel", "matthew", "anthony",
    "joshua", "andrew", "ryan", "jacob", "nicholas", "eric", "stephen",
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
    "donna", "michelle", "carol", "amanda", "melissa", "deborah",
    "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen",
    "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma",
    "nicole", "helen", "samantha", "katherine", "christine", "debra",
    "rachel", "carolyn", "janet", "catherine", "maria", "heather",
    "diane", "julie", "joyce", "victoria", "kelly", "christina",
    "joan", "evelyn", "lauren", "judith", "olivia", "alice", "julia",
    "ryan", "brandon", "adam", "tyler", "zachary", "austin", "kevin",
    "jason", "jeff", "gary", "timothy", "jose", "larry", "jeffrey",
    "frank", "scott", "eric", "stephen", "raymond", "gregory", "samuel",
    "benjamin", "patrick", "jack", "dennis", "jerry", "walter", "peter",
    "henry", "harold", "douglas", "arthur", "lawrence", "roger",
    "alex", "steve", "charlie", "mike", "josh", "chris", "tom", "ben",
    "jake", "lily", "bob", "mac",
    # ── Common Filipino first names ──────────────────────────────────────
    "juan", "pedro", "jose", "maria", "ana", "rosa", "luz", "grace",
    "amor", "joy", "faith", "hope", "carlo", "miguel", "angelo",
    "danilo", "mario", "mario", "rodrigo", "ferdinand", "imelda",
    "nena", "neneng", "lita", "nora", "vilma", "sharon", "maricel",
    "edgar", "ernesto", "renato", "rolando", "romeo", "eduardo",
    # ── Countries ────────────────────────────────────────────────────────
    "philippines", "america", "usa", "uk", "china", "japan", "korea",
    "india", "russia", "france", "germany", "italy", "spain", "brazil",
    "canada", "australia", "mexico", "indonesia", "thailand", "vietnam",
    "singapore", "malaysia", "taiwan", "hongkong", "israel", "iran",
    "iraq", "ukraine", "turkey", "egypt", "nigeria", "kenya", "ghana",
    "pakistan", "bangladesh", "srilanka", "nepal", "myanmar", "cambodia",
    "laos", "brunei", "timor", "argentina", "colombia", "peru", "chile",
    "venezuela", "cuba", "haiti", "jamaica", "panama", "sweden", "norway",
    "denmark", "finland", "netherlands", "belgium", "switzerland",
    "austria", "portugal", "greece", "poland", "czechia", "hungary",
    "romania", "bulgaria", "serbia", "croatia", "slovakia", "ukraine",
    "newzealand", "southafrica", "morocco", "ethiopia", "somalia",
    # ── Cities / places ──────────────────────────────────────────────────
    "manila", "cebu", "davao", "quezon", "makati", "pasig", "taguig",
    "london", "paris", "berlin", "rome", "madrid", "tokyo", "beijing",
    "shanghai", "delhi", "mumbai", "sydney", "toronto", "chicago",
    "houston", "phoenix", "losangeles", "newyork", "boston", "seattle",
    "miami", "dallas", "denver", "atlanta", "lasvegas", "singapore",
    "dubai", "istanbul", "moscow", "amsterdam", "barcelona", "vienna",
    "brussels", "stockholm", "oslo", "copenhagen", "zurich", "prague",
    "warsaw", "budapest", "athens", "lisbon", "dublin", "edinburgh",
    "cairo", "nairobi", "lagos", "johannesburg", "casablanca",
    # ── Continents / major regions ───────────────────────────────────────
    "asia", "europe", "africa", "america", "oceania", "antarctica",
    "middleeast", "caribbean", "scandinavia", "balkans",
    # ── Religions / religious institutions ──────────────────────────────
    "christianity", "islam", "hinduism", "buddhism", "judaism",
    "catholicism", "protestantism", "baptist", "mormon", "scientology",
    "quaker", "methodist", "lutheran", "calvinist", "jehovah",
    "vatican", "mosque", "synagogue", "temple",
    # ── Celebrities / public figures (non-political) ─────────────────────
    "oprah", "ellen", "beyonce", "rihanna", "adele", "taylor", "swift",
    "bieber", "kardashian", "jenner", "kanye", "drake", "eminem",
    "madonna", "britney", "spears", "shakira", "ariana", "grande",
    "elon", "musk", "bezos", "zuckerberg", "gates", "buffett", "jobs",
    "einstein", "newton", "darwin", "freud", "socrates", "plato",
    "aristotle", "shakespeare", "tolkien", "rowling", "spielberg",
    "scorsese", "tarantino", "kubrick", "chaplin", "eastwood",
    "pacquiao", "lebron", "jordan", "messi", "ronaldo", "neymar",
    "federer", "nadal", "djokovic", "tyson", "ali", "mayweather",
    "isis",
}

AAC_SEED_PHRASES = [
    "i need help",
    "i need water",
    "i need food",
    "i need medicine",
    "i am hungry",
    "i am thirsty",
    "i am tired",
    "i feel pain",
    "i feel cold",
    "i feel hot",
    "please help me",
    "please call my family",
    "please give me medicine",
    "can you help me",
    "can you move me",
    "can you turn on the light",
    "can you turn off the light",
    "i want to rest",
    "i want to sleep",
    "i want to go home",
    "thank you",
    "yes please",
    "no thank you",
]

# Vulgar / sexually explicit / crude English words
ENGLISH_VULGAR = {
    # Strong profanity
    "fuck", "fucking", "fucked", "fucker", "fucks",
    "shit", "shitty", "bullshit", "horseshit",
    "bitch", "bitches", "bastard", "bastards",
    "ass", "asses", "asshole", "assholes",
    "damn", "damned", "goddamn",
    "crap", "crappy", "cunt", "cunts",
    "dick", "dicks", "cock", "cocks", "prick",
    "pussy", "pussies", "whore", "whores", "slut", "slutty",
    "nigger", "nigga", "chink", "spic", "kike", "faggot", "dyke",
    # Explicit sexual
    "porn", "porno", "sex", "sexy", "horny", "nude", "naked",
    "penis", "vagina", "boobs", "breast", "nipple", "orgasm",
    "masturbate", "ejaculate", "erect", "aroused",
}

# Filipino/Tagalog words that don't belong in an English vocabulary
ENGLISH_FILIPINO_WORDS = {
    # Pronouns & particles
    "ako", "ikaw", "siya", "tayo", "kami", "kayo", "sila",
    "ito", "iyon", "iyan", "dito", "doon", "diyan",
    "na", "ba", "pa", "nga", "po", "ho", "din", "rin", "lang", "naman",
    "kasi", "kung", "kahit", "para", "pero", "at", "ay", "ni", "ng",
    "nang", "sa", "kay", "pag", "kapag", "habang", "dahil",
    # Common words
    "hindi", "wala", "may", "mayroon", "yung", "yun", "diba",
    "bakit", "paano", "sana", "talaga", "grabe", "nandito",
    "nandoon", "ngayon", "kahapon", "bukas", "mamaya", "kanina",
    "gutom", "uhaw", "tulog", "gising", "pagkain", "tubig",
    "bahay", "trabaho", "paaralan", "ospital", "tindahan",
    "kumain", "uminom", "matulog", "pumunta", "bumalik",
    "masaya", "malungkot", "maganda", "pangit", "mahal", "mura",
    "malaki", "maliit", "bago", "luma", "mainit", "malamig",
    "salamat", "pakiusap", "paumanhin", "sandali", "halika",
    "siguro", "medyo", "sobra", "masyado", "konti", "marami",
    "oo", "huwag", "pwede", "kailangan", "gusto", "ayaw",
}

# Combined English blocklist
ENGLISH_BLOCKLIST = ENGLISH_PROPER_NAMES | ENGLISH_VULGAR | ENGLISH_FILIPINO_WORDS

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
    "np": "no problem", "ty": "thank you", "tq": "thank you", "tysm": "thank you so much",
    "thx": "thanks", "thnx": "thanks", "tha": "thank", "thk": "thank",
    "yw": "you're welcome", "wb": "welcome back",
    "ofc": "of course", "nbd": "no big deal", "def": "definitely",
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


def clean_token(token: str) -> str:
    return re.sub(r"^[^a-zA-Z']+|[^a-zA-Z']+$", "", token.lower())


def phrase_tokens(phrase: str) -> list:
    return [cleaned for token in phrase.split() if (cleaned := clean_token(token))]


def contains_blocked_token(tokens) -> bool:
    return any(token in ENGLISH_BLOCKLIST for token in tokens)


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
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
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
                    token_key = clean_token(token)
                    score = r["score"]
                    if score < MIN_SCORE or not is_valid_token(token) or token_key in ENGLISH_BLOCKLIST:
                        continue
                    category_words[category][token] += score
                    phrase = re.sub(r'\s+', ' ', tmpl.replace("<mask>", token)).strip().rstrip('.')
                    if not contains_blocked_token(phrase_tokens(phrase)):
                        all_corpus_seeds.append(phrase.lower())
            except Exception:
                pass

    print(f"\r    [{'█'*40}] {total_templates}/{total_templates} ✓\n")

    all_corpus_seeds.extend(AAC_SEED_PHRASES)

    print("🔧 Expanding verb forms...")
    for base in list(category_words.get("verbs", Counter()).keys())[:40]:
        for variant in expand_verb_forms(base):
            if is_valid_token(variant) and variant not in category_words["verbs"]:
                category_words["verbs"][variant] += 0.001
    print("✓ Done.\n")

    # ── Post-process vocabulary ───────────────────────────────────────────────
    print("🔎 Filtering vocabulary (politicians, vulgar, Filipino words)...")
    vocabulary     = {}
    all_words_flat = set()
    removed        = []
    for cat, counter in category_words.items():
        mapped = CATEGORY_MAP.get(cat, cat)
        if mapped not in vocabulary:
            vocabulary[mapped] = []
        for w in top_words(counter, n=100):
            if w in ENGLISH_BLOCKLIST:
                removed.append(w)
                continue
            if w not in all_words_flat:
                vocabulary[mapped].append(w)
                all_words_flat.add(w)
    if removed:
        print(f"   ✗ Removed {len(removed)} blocked words: {', '.join(sorted(set(removed))[:20])}"
              + (" ..." if len(set(removed)) > 20 else ""))

    # ── communication_corpus: deduplicated phrase strings (legacy / display)
    seen_phrases = set()
    corpus = []
    for phrase in all_corpus_seeds:
        clean = phrase.strip()
        tokens = phrase_tokens(clean)
        if clean not in seen_phrases and len(tokens) >= 2 and not contains_blocked_token(tokens):
            corpus.append(clean)
            seen_phrases.add(clean)
        if len(corpus) >= 1600:
            break

    # ── corpus_sequences: tokenised lists — used by model.py for n-gram training
    # Each entry is a list of word strings preserving sequential order so that
    # bigram/trigram co-occurrence counts reflect real phrase structure.
    corpus_sequences = []
    seen_seq = set()
    for phrase in all_corpus_seeds:
        tokens = phrase_tokens(phrase)
        if len(tokens) >= 2 and not contains_blocked_token(tokens):
            key = " ".join(tokens)
            if key not in seen_seq:
                corpus_sequences.append(tokens)
                seen_seq.add(key)
        if len(corpus_sequences) >= 2000:
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
