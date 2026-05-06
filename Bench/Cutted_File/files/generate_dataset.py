"""
generate_dataset.py — Filipino Dataset Generator using RoBERTa-Tagalog

Generates filipino_dataset.json from scratch using:
  jcblaise/roberta-tagalog-base (masked language model)

Changes from previous version:
  - Added `corpus_sequences` field: full tokenised phrase lists for n-gram
    training (preserves word-order context, unlike the flat vocabulary list)
  - CATEGORY_MAP now includes slang_taglish -> slang
  - generate_if_missing() helper for main.py integration

Can be run standalone:
    python generate_dataset.py

Or called from main.py on first startup via generate_if_missing().
"""

from collections import defaultdict, Counter
import json
import re
from datetime import date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME  = "jcblaise/roberta-tagalog-base"
TOP_K       = 50
MIN_SCORE   = 0.003
OUTPUT_FILE = "filipino_dataset.json"

# ─────────────────────────────────────────────
# SEED TEMPLATES
# ─────────────────────────────────────────────
TEMPLATES = {
    "pronouns": [
        "<mask> ay gutom na.",
        "<mask> ay natutulog.",
        "Kumain na ba <mask>?",
        "Nasaan si <mask>?",
        "<mask> ay masaya ngayon.",
        "Punta na <mask> sa bahay.",
        "Kasama ko si <mask>.",
        "Para kay <mask> ito.",
        "Sinabi ni <mask> na pumunta na.",
        "Hinahanap ko si <mask>.",
        "Ang <mask> ay magaling.",
        "Nandito na si <mask>.",
    ],
    "verbs": [
        "Gusto kong <mask> ng pagkain.",
        "Kailangan niya <mask> ng tubig.",
        "Siya ay <mask> sa kwarto.",
        "Maaari ba akong <mask> dito?",
        "Pwede ka bang <mask> mamaya?",
        "Ayoko nang <mask> ngayon.",
        "Nais niyang <mask> ng maaga.",
        "Nagsimula na siyang <mask>.",
        "Patuloy siyang <mask> araw-araw.",
        "Kailangan nating <mask> ng mabilis.",
        "Mag-<mask> ka na.",
        "Tulungan mo akong <mask>.",
        "Hindi ko gustong <mask> ngayon.",
        "Sana ay <mask> na tayo.",
        "Oras na para <mask>.",
        "Puwede ka nang <mask> ngayon.",
        "Subukan nating <mask> bukas.",
        "Huwag nang <mask> pa.",
        "Kaya mo bang <mask> ngayon?",
        "Sino ang <mask> doon?",
        "Bakit ka <mask> dito?",
        "Anong oras ka <mask>?",
    ],
    "adjectives": [
        "Ang pagkain ay <mask> talaga.",
        "Ang bata ay <mask> kahapon.",
        "Pakiramdam ko ay <mask> ngayon.",
        "Ang gawa mo ay <mask> na.",
        "Ang lugar ay <mask> at tahimik.",
        "Sobrang <mask> ng panahon ngayon.",
        "Ang kanyang mukha ay <mask>.",
        "Ang tubig ay <mask> ngayon.",
        "Naramdaman niyang <mask> ang katawan niya.",
        "Ang boses niya ay <mask>.",
        "Ang damit niya ay <mask>.",
        "Ang pagkain nila ay <mask>.",
        "Ang kwarto ay <mask> at malinis.",
        "Ang trabaho niya ay <mask>.",
        "Ang kanyang ugali ay <mask>.",
        "Ang presyo nito ay <mask>.",
        "Ang daan ay <mask> at maputik.",
    ],
    "nouns_people": [
        "Ang aking <mask> ay nandito.",
        "Pumunta ang <mask> sa ospital.",
        "Sinabi ng <mask> na kumain na.",
        "Hinahanap ko ang aking <mask>.",
        "Ang <mask> ko ay magaling.",
        "Kasama ko ang aking <mask>.",
        "Nagtatrabaho ang <mask> sa opisina.",
        "Tinawagan ng <mask> si nanay.",
        "Ang <mask> namin ay maingat.",
        "Dumating na ang <mask>.",
        "Tinanong ng <mask> ang guro.",
        "Pumayag ang <mask> sa kahilingan.",
        "Ang <mask> ang nag-alaga sa kanya.",
    ],
    "nouns_places": [
        "Pumunta siya sa <mask>.",
        "Nandoon siya sa <mask>.",
        "Kailangan naming pumunta sa <mask>.",
        "Malapit lang ang <mask> dito.",
        "Naghihintay kami sa <mask>.",
        "Lalabas ako sa <mask> mamaya.",
        "Malayo ang <mask> mula dito.",
        "Nagtayo sila ng <mask> sa tabi.",
        "Maganda ang <mask> sa hapon.",
        "Maraming tao sa <mask> ngayon.",
        "Nanggaling siya sa <mask>.",
        "Mahal ang pagkain sa <mask>.",
        "Malapit na tayong makarating sa <mask>.",
    ],
    "nouns_things": [
        "Kailangan ko ng <mask>.",
        "Nasaan ang aking <mask>?",
        "Ibigay mo sa akin ang <mask>.",
        "Gumagamit siya ng <mask>.",
        "Nawala ang aking <mask>.",
        "Bago ang kanyang <mask>.",
        "Sira na ang <mask> namin.",
        "Bumili siya ng <mask> kahapon.",
        "Mahal ang <mask> ngayon.",
        "Maaari bang hiramin ang <mask>?",
        "Ilagay mo ang <mask> doon.",
        "Dala ko ang <mask> ko.",
        "Wala na kaming <mask>.",
        "Kailangan naming ayusin ang <mask>.",
    ],
    "food_drink": [
        "Gusto kong kumain ng <mask>.",
        "Masarap ang <mask> dito.",
        "Nag-order siya ng <mask>.",
        "Nais niyang uminom ng <mask>.",
        "Ang paboritong ulam ko ay <mask>.",
        "Luto niya ng <mask> para sa hapunan.",
        "Walang <mask> sa ref.",
        "Bumili kami ng <mask> sa tindahan.",
        "Gutom na ako, gusto ko ng <mask>.",
        "Masustansya ang <mask> para sa katawan.",
        "Ang almusal namin ay <mask>.",
        "Ihanda mo ang <mask> para sa bisita.",
        "Anong <mask> ang gusto mo?",
        "Paborito ng mga bata ang <mask>.",
        "Ang <mask> ay mainit pa.",
        "Uhaw na ako, pakiabot ng <mask>.",
    ],
    "time": [
        "Pupunta kami <mask>.",
        "Gagawin natin ito <mask>.",
        "Nagsimula siya noong <mask>.",
        "Aabangan kita <mask>.",
        "Bumalik siya <mask>.",
        "Magkikita tayo <mask>.",
        "Maaga pa nang <mask>.",
        "Tapos na ang <mask>.",
        "Naghihintay na kami mula <mask>.",
        "Simula <mask> ay nagbago na siya.",
        "Hanggang <mask> lang tayo.",
        "Ano ang plano mo sa <mask>?",
    ],
    "particles": [
        "Sige <mask> pumunta na tayo.",
        "Gusto ko <mask> kumain.",
        "Hindi <mask> ako pupunta.",
        "Pwede <mask> ba tayo?",
        "Kailangan <mask> umuwi.",
        "Tama <mask> ang sinabi mo.",
        "Ganyan <mask> talaga siya.",
        "Mahal kita <mask>.",
        "Okay <mask> ang lahat.",
        "Tara <mask> umalis na.",
        "Wag <mask> malungkot.",
        "Sama <mask> tayo.",
        "Punta <mask> tayo bukas.",
        "Kain <mask> muna tayo.",
    ],
    "questions": [
        "<mask> ang pangalan mo?",
        "<mask> siya pumunta?",
        "<mask> ba ang nangyari?",
        "<mask> mo ito ginawa?",
        "<mask> na tayo uuwi?",
        "<mask> ang gusto mo?",
        "<mask> ang problema?",
        "<mask> siya ngayon?",
        "<mask> ka pumunta doon?",
        "<mask> na ang susunod?",
        "<mask> ang maganda dito?",
        "<mask> ang kailangan mo?",
    ],
    "expressions": [
        "<mask>, salamat sa tulong mo.",
        "<mask> hindi ko alam.",
        "<mask> na lang.",
        "<mask>, okay lang yan.",
        "<mask> na tayo.",
        "Ang sagot ko ay <mask>.",
        "Sabi ko <mask> na.",
        "<mask>, ganyan talaga.",
        "Hindi ko <mask> iyon.",
        "Alam ko na, <mask>.",
        "<mask> naman oh.",
        "Ano ba <mask>.",
    ],
    "location": [
        "Nandito lang ako <mask>.",
        "Lumipat siya <mask>.",
        "Pumunta kami <mask> ng bahay.",
        "Naghihintay siya <mask>.",
        "Hanapin mo <mask>.",
        "Ilagay mo <mask>.",
        "Nandoon ang <mask>.",
        "Mula <mask> hanggang doon.",
        "Pumunta ka <mask>.",
        "Nasa <mask> ang lahat.",
    ],
    "body": [
        "Masakit ang aking <mask>.",
        "Nahawakan niya ang <mask> ko.",
        "Inayos niya ang <mask> niya.",
        "Naiinit ang <mask> ko.",
        "Naramdaman ng <mask> ko ang sakit.",
        "Napagod ang <mask> ko.",
        "Ang <mask> niya ay malusog.",
        "Sinugatan ang <mask> niya.",
        "Piniga niya ang <mask> niya.",
        "Malakas ang <mask> niya.",
    ],
    "emotions": [
        "Naramdaman niyang <mask> ang kanyang puso.",
        "Sobrang <mask> ng loob niya.",
        "Hindi niya mapigilan ang pagiging <mask>.",
        "Ang <mask> niya ay halata.",
        "Pakiramdam niya ay <mask> siya.",
        "Naging <mask> siya nang marinig iyon.",
        "Ang <mask> niya ay nakikita sa mukha.",
        "Nagtago siya dahil sa <mask>.",
        "Hindi niya matanggap ang <mask>.",
        "Puno ng <mask> ang kanyang puso.",
        "Ang <mask> ay hindi madaling tiisin.",
    ],
    "slang_taglish": [
        "Grabe <mask> talaga siya.",
        "Super <mask> ang nangyari.",
        "Legit na <mask> yan.",
        "Para sakin <mask> lang yon.",
        "Feeling ko <mask> na.",
        "Bet ko ang <mask> mo.",
        "Sana all ay <mask>.",
        "True <mask> yan.",
        "Char, <mask> lang naman.",
        "Charot, <mask> ko lang.",
        "Ang <mask> mo talaga.",
        "Solid <mask> yan.",
        "Grabe ang <mask> niya.",
        "Totoo bang <mask> yan?",
        "Ano ba <mask> mo?",
        "Wag kang <mask> dito.",
        "Ikaw talaga, laging <mask>.",
        "Nag-<mask> ka na naman.",
    ],
    "health_needs": [
        "Kailangan ko ng <mask> ngayon.",
        "Masakit ang aking <mask> ngayon.",
        "Nangangailangan ako ng <mask>.",
        "Pakiusap bigyan mo ako ng <mask>.",
        "Maaari ba akong kumuha ng <mask>?",
        "Walang <mask> dito.",
        "Kailangan ko ng tulong sa <mask>.",
        "Ang <mask> ko ay hindi maganda.",
        "Pumunta na tayo sa <mask>.",
        "Humingi siya ng <mask> sa doktor.",
        "Iniinom niya ang <mask> araw-araw.",
        "Kailangan niya ng <mask> para gumaling.",
        "Nagkaroon siya ng <mask> kahapon.",
        "Ang <mask> ay pangontra sa sakit.",
    ],
    "communication": [
        "Sabihin mo sa kanya na <mask>.",
        "Pakitawagan ang <mask>.",
        "Gusto kong makipag-usap sa <mask>.",
        "Ipaliwanag mo sa akin ang <mask>.",
        "Paki-text ako ng <mask>.",
        "Pakisend sa akin ang <mask>.",
        "Sinabi niya na <mask>.",
        "Tinanong niya kung <mask>.",
        "Ipinaabot niya ang <mask>.",
        "Nag-message siya tungkol sa <mask>.",
        "Ibalita mo sa kanya ang <mask>.",
        "Pakisabi kay <mask> na pumunta na.",
    ],
    "school_work": [
        "May <mask> kami bukas.",
        "Kailangan naming gawin ang <mask>.",
        "Ang <mask> namin ay mahirap.",
        "Nag-aaral siya ng <mask>.",
        "Ang guro namin sa <mask> ay mahigpit.",
        "Pasa na ang <mask> ko.",
        "May <mask> pa akong gagawin.",
        "Ang <mask> ko ay bukas na ang deadline.",
        "Nahirapan siya sa <mask>.",
        "Magaling siya sa <mask>.",
        "Ang <mask> namin ay maaga.",
        "Binigyan kami ng <mask> ng titser.",
    ],
    "daily_life": [
        "Tuwing umaga, <mask> muna siya.",
        "Bago matulog, <mask> muna siya.",
        "Pagkatapos kumain, <mask> siya.",
        "Habang naghihintay, <mask> siya.",
        "Sa bahay, madalas siyang <mask>.",
        "Tuwing gabi, <mask> sila.",
        "Pagdating sa trabaho, agad siyang <mask>.",
        "Sa paglalaro, <mask> ang bata.",
        "Kapag bakasyon, <mask> kami.",
        "Sa daan, <mask> siya.",
    ],
}

CATEGORY_MAP = {
    "emotions":      "expressions",
    "slang_taglish": "slang",
    "health_needs":  "expressions",
    "communication": "expressions",
    "school_work":   "nouns_things",
    "daily_life":    "verbs",
}

AAC_SEED_PHRASES = [
    "kailangan ko ng tulong",
    "kailangan ko ng tubig",
    "kailangan ko ng pagkain",
    "kailangan ko ng gamot",
    "gutom na ako",
    "uhaw na ako",
    "pagod na ako",
    "masakit ang katawan ko",
    "masakit ang ulo ko",
    "nilalamig ako",
    "naiinitan ako",
    "tulungan mo ako",
    "pakiusap tulungan mo ako",
    "pakitawagan ang pamilya ko",
    "pakibigay ang gamot ko",
    "pwede mo ba akong tulungan",
    "pwede mo ba akong ilipat",
    "pakibukas ang ilaw",
    "pakipatay ang ilaw",
    "gusto kong magpahinga",
    "gusto kong matulog",
    "gusto kong umuwi",
    "salamat po",
    "oo po",
    "hindi po",
]

# ─────────────────────────────────────────────
# FILTER SETS — applied after vocabulary generation
# ─────────────────────────────────────────────

# Philippine politicians, public figures, celebrities
FILIPINO_POLITICIANS = {
    # Presidents & families
    "duterte", "digong", "marcos", "bongbong", "imee", "aquino", "noynoy",
    "cory", "arroyo", "gloria", "estrada", "erap", "ramos", "pnoy",
    # Senators / officials
    "trillanes", "cayetano", "pimentel", "drilon", "lacson", "sotto",
    "gordon", "villar", "ejercito", "binay", "revilla", "enrile",
    "santiago", "miriam", "poe", "roxas", "panelo", "tulfo", "faeldon",
    "lozada", "leni", "robredo", "sara", "pacquiao", "manny", "pacman",
    # Other known names the model predicts
    "kris", "robin", "marian", "sheryl", "david", "julius",
    # Foreign politicians
    "trump", "obama", "biden", "putin", "zelensky", "xi", "jinping",
    "modi", "macron", "trudeau", "sunak", "scholz", "kishida", "kim",
}

# Common first/last names and religious figures the model predicts
# in person slots — not appropriate vocabulary words
FILIPINO_PROPER_NAMES = {
    # Biblical / religious figures
    "jesus", "hesus", "kristo", "maria", "jose", "diyos", "dios",
    "panginoon", "lord", "satanas", "lucifer", "biblia", "bibliya",
    "papa", "papa", "francis", "pedro", "pablo", "juan", "david",
    "elias", "cesar", "nelson", "gordon", "tom", "tomas", "robin",
    # Common Filipino first names the model predicts
    "ana", "rosa", "luz", "grace", "amor", "joy", "faith", "hope",
    "mark", "john", "paul", "james", "luke", "matthew",
    "tony", "carlo", "carlo", "miguel", "angelo", "danilo",
    "nena", "nene", "lola", "lolo", "tita", "tito", "kuya", "ate",
    # Celebrities / non-political public figures
    "pacman", "marian", "sheryl", "kris",
}

# Vulgar / sexually explicit / crude Filipino words
FILIPINO_VULGAR = {
    # Strong profanity
    "putang", "puta", "putangina", "tangina", "gago", "gaga", "tanga",
    "ulol", "leche", "letse", "hinayupak", "hayop", "kupal",
    "tarantado", "pakyu", "punyeta", "syet", "bwisit", "lintik",
    # Explicit sexual / body slang
    "titi", "puki", "ari", "pepe", "kepyas", "burat", "etits", "otin",
    "suso", "pwet", "puwet", "puet", "tae", "ihi", "libog", "malibog",
    "kantot", "jakol", "bastos", "malaswa", "mahalay", "hubad",
    "hubaran", "hubo", "seksi",
}

# Words misclassified into the wrong category by the model
# (not wrong words per se, but they pollute categories like pronouns/time)
FILIPINO_MISCLASSIFIED = {
    # Non-pronouns that appear in pronouns category
    "resulta", "pagkatapos", "tubig", "mundo", "makina", "ngayon",
    "doktor", "problema", "sagot", "kalikasan", "guro", "libro",
    "buhay", "isa", "trainer", "kita", "pagkain", "lang",
    "babae", "hari", "asin", "kabayo",
    # Non-time words in time category
    "digmaan", "laban", "hinaharap", "eleksyon", "kampanya", "halalan",
    "labanan", "ministeryo", "proseso", "patay", "hospice", "giyera",
    "palabas", "pagluluto", "pagkabata",
    # Non-particles in particles category
    "itan", "ip", "ing", "umin", "dak", "ta",
    # Non-food in food category
    "dasal", "paumanhin", "kinakailangan", "kuryente", "kulay", "ilaw",
    "klase", "tulog", "tawad", "mensahe", "kanta", "sorry", "bagay",
    "temperatura", "taglamig", "pasko", "musika", "payo", "kaunti",
    # Non-location in location category
    "panganib", "impiyerno", "lindol", "baha", "apoy", "sunog", "ulan",
    "sya", "bigla",
    # Non-adjective in adjectives category
    "namatay", "ipinanganak", "pumanaw", "bumagsak", "tumaas",
    "matematika", "dumadaloy", "pataas", "gumalaw", "isinilang",
    "nagbabago", "namamaga", "nanginginig", "bumigay", "ginto",
    # Non-verb particles that slip into verbs
    "na", "ba", "rin", "pa", "wala", "dito", "rito", "nandito",
    "nandoon", "naroon", "talaga", "aral", "ingat",
}

# Pure English words that don't belong in Filipino vocabulary
# (Taglish slang intentionally kept: grabe, bet, solid, enjoy, etc.)
FILIPINO_ENGLISH_WORDS = {
    "the", "and", "but", "for", "with", "from", "that", "this", "they",
    "have", "not", "are", "was", "were", "been", "will", "would", "could",
    "should", "may", "can", "did", "does", "had", "has", "its", "our",
    "their", "your", "his", "her", "him", "them", "who", "what", "where",
    "when", "why", "how", "all", "any", "some", "one", "two", "three",
    "more", "most", "also", "just", "then", "than", "into", "over",
    "after", "before", "about", "out", "up", "down", "off", "on", "in",
    "at", "by", "to", "of", "is", "it", "if", "or", "do", "go", "get",
    "got", "put", "let", "say", "said", "see", "saw", "come", "came",
    "take", "make", "know", "want", "use", "find", "give", "work", "call",
    "need", "feel", "seem", "keep", "try", "ask", "turn", "start", "show",
    "hear", "play", "run", "move", "live", "hold", "believe", "happen",
    "include", "continue", "become", "follow", "stop", "create", "speak",
    "read", "spend", "grow", "open", "walk", "win", "offer", "remember",
    "love", "consider", "appear", "buy", "wait", "plan", "learn", "change",
    "lead", "watch", "form", "help", "develop", "carry", "break", "cover",
    "build", "stay", "fall", "reach", "kill", "remain", "suggest", "raise",
    "pass", "sell", "require", "report", "decide", "pull", "place", "add",
    "yes", "no", "hi", "hello", "bye", "good", "bad", "big", "old",
    "new", "high", "low", "long", "right", "left", "next", "last", "many",
    "much", "few", "well", "even", "back", "only", "same", "own", "every",
    "house", "home", "school", "work", "time", "year", "day", "week",
    "month", "night", "morning", "life", "hand", "part", "place", "case",
    "company", "system", "program", "question", "government",
    "country", "city", "water", "world", "family", "man", "woman", "child",
    "way", "thing", "person", "people", "room", "money", "area", "book",
    "eye", "number", "face", "door", "car", "food", "body", "music",
    "story", "fact", "idea", "game", "name", "group", "problem", "side",
    "point", "small", "large", "white", "black", "real", "true",
    "important", "possible", "public", "private", "free", "able", "late",
    "hard", "early", "major", "second", "third",
}

# Combined Filipino blocklist
# NOTE: FILIPINO_MISCLASSIFIED is defined above for reference and logging
# but intentionally NOT included here — those are valid Filipino words that
# the model placed in the wrong category. They are kept in the vocabulary
# so they can still appear under a correct category elsewhere.
FILIPINO_BLOCKLIST = (
    FILIPINO_POLITICIANS
    | FILIPINO_PROPER_NAMES
    | FILIPINO_VULGAR
    | FILIPINO_ENGLISH_WORDS
)

SHORTCUTS = {
    "lng": "lang", "nlng": "nalang", "nmn": "naman",
    "ksi": "kasi", "kse": "kasi", "kng": "kung", "khit": "kahit",
    "d2": "dito", "dn": "doon", "dun": "doon",
    "pde": "pwede", "pwd": "pwede",
    "sna": "sana", "tlg": "talaga", "tlga": "talaga",
    "sya": "siya", "xa": "siya", "aq": "ako", "q": "ako",
    "nde": "hindi", "hnd": "hindi", "di": "hindi",
    "pra": "para", "wla": "wala", "my": "may",
    "ung": "yung", "un": "yun", "dba": "diba", "db": "diba",
    "bat": "bakit", "bkt": "bakit", "pno": "paano", "pano": "paano",
    "tyo": "tayo", "kyo": "kayo", "kmi": "kami", "cla": "sila",
    "ikw": "ikaw", "grbe": "grabe",
    "cguro": "siguro", "sguro": "siguro",
    "nman": "naman", "mdyo": "medyo",
    "wag": "huwag", "hwag": "huwag",
    "thnk": "salamat", "ty": "salamat", "tnx": "salamat", "slmat": "salamat", "slm": "salamat",
    "pls": "pakiusap", "plz": "pakiusap", "plss": "pakiusap",
    "ok": "okay", "k": "okay", "sge": "sige",
    "lol": "nakakatawa",
    "omg": "grabe", "wtf": "ano ba",
    "brb": "babalik agad", "afk": "wala sandali",
    "g": "sige", "gg": "sige na",
    "fr": "totoo", "frfr": "totoo talaga",
    "ngl": "totoo lang", "tbh": "totoo lang",
    "imo": "sa tingin ko",
    "btw": "by the way", "fyi": "alam mo ba",
    "asap": "agad-agad", "rn": "ngayon",
    "hbd": "maligayang bati", "hny": "maligayang bagong taon",
    "gm": "magandang umaga", "gn": "magandang gabi",
    "ga": "magandang araw", "gh": "magandang hapon",
    "tc": "mag-ingat", "imy": "namimiss kita",
    "ily": "mahal kita",
    "bff": "pinakamatalik na kaibigan",
    "smh": "nakakahiya talaga", "irl": "sa totoo",
    "dm": "direktang mensahe", "pm": "personal na mensahe",
    "np": "walang anuman", "nvm": "wag na",
    "ikr": "totoo nga", "tbf": "sa totoo lang",
    "ano": "ano", "eh": "eh", "ndi": "hindi", "nd": "hindi",
    "mkta": "makita", "mkikta": "makikita", "mgkta": "magkita",
    "kpg": "kapag", "pag": "kapag", "pgka": "pagkatapos",
    "dpat": "dapat", "dpt": "dapat",
    "nsaan": "nasaan", "san": "nasaan",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_valid_token(token: str) -> bool:
    token = token.strip()
    if len(token) < 2:
        return False
    if not re.match(r'^[a-zA-ZÑñ\-]+$', token):
        return False
    return True


def clean_token(token: str) -> str:
    return re.sub(r'^[^a-zA-ZÑñ]+|[^a-zA-ZÑñ]+$', '', token.lower())


def phrase_tokens(phrase: str) -> list:
    return [cleaned for token in phrase.split() if (cleaned := clean_token(token))]


def contains_blocked_token(tokens) -> bool:
    return any(token in FILIPINO_BLOCKLIST for token in tokens)


def expand_affixes(word: str) -> list:
    variants = [word]
    variants += [f"mag{word}", f"nag{word}", f"magka{word}", f"nagka{word}"]
    if len(word) > 1 and word[0] in "bcdfghjklmnpqrstvwxyz":
        variants += [f"{word[0]}um{word[1:]}"]
    variants += [f"ma{word}", f"na{word}"]
    variants += [f"{word}an", f"{word}han"]
    variants += [f"{word}in", f"{word}hin"]
    variants += [f"i{word}"]
    if len(word) >= 2:
        variants += [f"{word[:2]}{word}"]
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
                    token_key = clean_token(token)
                    score = r["score"]
                    if score < MIN_SCORE or not is_valid_token(token) or token_key in FILIPINO_BLOCKLIST:
                        continue
                    category_words[category][token] += score
                    phrase = re.sub(r'\s+', ' ', tmpl.replace("<mask>", token)).strip().rstrip('.')
                    if not contains_blocked_token(phrase_tokens(phrase)):
                        all_corpus_seeds.append(phrase.lower())
            except Exception:
                pass

    print(f"\r    [{'█'*40}] {total_templates}/{total_templates} ✓\n")

    all_corpus_seeds.extend(AAC_SEED_PHRASES)

    print("🔧 Expanding verb affixes...")
    for base in list(category_words.get("verbs", Counter()).keys())[:40]:
        for variant in expand_affixes(base):
            if is_valid_token(variant) and variant not in category_words["verbs"]:
                category_words["verbs"][variant] += 0.001
    print("✓ Done.\n")

    # ── Post-process vocabulary ───────────────────────────────────────────────
    print("🔎 Filtering vocabulary (politicians, vulgar, English words)...")
    vocabulary     = {}
    all_words_flat = set()
    removed        = []
    for cat, counter in category_words.items():
        mapped = CATEGORY_MAP.get(cat, cat)
        if mapped not in vocabulary:
            vocabulary[mapped] = []
        for w in top_words(counter, n=100):
            if w in FILIPINO_BLOCKLIST:
                removed.append(w)
                continue
            if w not in all_words_flat:
                vocabulary[mapped].append(w)
                all_words_flat.add(w)
    if removed:
        print(f"   ✗ Removed {len(removed)} blocked words: {', '.join(sorted(set(removed))[:20])}"
              + (" ..." if len(set(removed)) > 20 else ""))

    # ── communication_corpus: deduplicated phrase strings (for display / legacy)
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
            "version":          "4.0-roberta",
            "language":         "Filipino-Tagalog (RoBERTa-generated)",
            "description":      "Auto-generated Dataset via jcblaise/roberta-tagalog-base — For Gaze Based Digital Keyboard",
            "total_words":      total_words,
            "total_phrases":    total_phrases,
            "total_shortcuts":  total_shortcuts,
            "last_updated":     str(date.today()),
            "model_used":       MODEL_NAME,
        },
        "vocabulary":           vocabulary,
        "shortcuts":            SHORTCUTS,
        "communication_corpus": corpus,
        # NEW — sequential token lists for n-gram training in model.py
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
