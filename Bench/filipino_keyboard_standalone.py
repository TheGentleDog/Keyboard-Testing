import tkinter as tk
from tkinter import ttk
from collections import defaultdict, Counter
import json
import os
import pickle

# Configuration
MAX_SUGGESTIONS = 8
NGRAM_CACHE_FILE = "ngram_model_standalone.pkl"
USER_LEARNING_FILE = "user_learning.json"
MODEL_VERSION = "2.0_with_corpus"  # Change this to force rebuild

# =============================================================================
# BUILT-IN FILIPINO VOCABULARY (NO EXTERNAL DATASETS NEEDED!)
# =============================================================================

# Common conversational Filipino words (~2000+ words)
FILIPINO_WORDS = [
    # Pronouns
    "ako", "ko", "akin", "ikaw", "ka", "mo", "iyo", "siya", "niya", "kaniya",
    "kami", "namin", "amin", "tayo", "natin", "atin", "kayo", "ninyo", "inyo",
    "sila", "nila", "kanila",
    
    # Common Verbs - Base + Conjugations
    "kain", "kumain", "kakain", "kumakain", "kinain",
    "inom", "uminom", "iinom", "umiinom", "ininom",
    "punta", "pumunta", "pupunta", "pumupunta",
    "uwi", "umuwi", "uuwi", "umuuwi",
    "lakad", "lumakad", "lalakad", "lumalakad",
    "takbo", "tumakbo", "tatakbo", "tumatakbo",
    "dating", "dumating", "darating", "dumarating",
    "alis", "umalis", "aalis", "umaalis",
    "tulog", "matulog", "matutulog", "natutulog", "natulog",
    "gising", "gumising", "gigising", "gumagising",
    "ligo", "maligo", "maliligo", "naliligo",
    "linis", "maglinis", "maglilinis", "naglilinis",
    "sabi", "sabihin", "sinabi", "nagsabi", "nagsasabi",
    "salita", "magsalita", "magsasalita", "nagsasalita",
    "tanong", "magtanong", "magtatanong", "nagtatanong",
    "sagot", "sumagot", "sasagot", "sumasagot",
    "aral", "mag-aral", "mag-aaral", "nag-aaral",
    "trabaho", "magtrabaho", "magtatrabaho", "nagtatrabaho",
    "basa", "magbasa", "magbabasa", "nagbabasa",
    "sulat", "magsulat", "magsusulat", "nagsusulat",
    "laro", "maglaro", "maglalaro", "naglalaro",
    "nood", "manood", "manonood", "nanonood",
    "kanta", "kumanta", "kakanta", "kumakanta",
    "sayaw", "sumayaw", "sasayaw", "sumasayaw",
    "bili", "bumili", "bibili", "bumibili",
    "bayad", "magbayad", "magbabayad", "nagbabayad",
    "tuwa", "matuwa", "matutuwa", "natutuwa",
    "iyak", "umiyak", "iiyak", "umiiyak",
    "tawa", "tumawa", "tatawa", "tumatawa",
    "galit", "magalit", "magagalit", "nagagalit",
    "luto", "magluto", "magluluto", "nagluluto",
    "hugas", "maghugas", "maghuhugas", "naghuhugas",
    "hintay", "maghintay", "maghihintay", "naghihintay",
    "hanap", "maghanap", "maghahanap", "naghahanap",
    "bigay", "magbigay", "magbibigay", "nagbibigay",
    "kuha", "kumuha", "kukuha", "kumukuha",
    "gamit", "gumamit", "gagamit", "gumagamit",
    
    # Adjectives
    "maganda", "ganda", "pangit", "pogi", "guwapo", "gwapo",
    "malaki", "laki", "maliit", "liit", "mataba", "taba", "payat",
    "mataas", "taas", "mababa", "baba",
    "mabuti", "buti", "masama", "sama", "magaling", "galing", "mahusay", "husay",
    "mabilis", "bilis", "mabagal", "bagal",
    "mainit", "init", "malamig", "lamig",
    "masarap", "sarap", "mapait", "pait", "matamis", "tamis", "maasim", "asim", "maalat", "alat",
    "masaya", "saya", "malungkot", "lungkot",
    "gutom", "nagugutom", "busog", "nabusog",
    "uhaw", "nauuhaw", "pagod", "napagod", "antok", "inaantok",
    "mahirap", "hirap", "madali", "dali", "mahal", "mura",
    
    # People
    "tao", "lalaki", "babae", "bata", "sanggol", "baby",
    "pamilya", "angkan", "ina", "nanay", "mama", "mommy", "inay",
    "ama", "tatay", "papa", "daddy", "itay",
    "lolo", "lola", "kuya", "ate", "bunso", "panganay",
    "kapatid", "tito", "tita", "tiyuhin", "tiyahin", "pamangkin", "pinsan",
    "asawa", "misis", "mister",
    "kaibigan", "barkada", "tropa", "kapitbahay", "kasama", "kasamahan",
    "guro", "titser", "maestra", "maestro", "estudyante", "mag-aaral",
    "doktor", "manggagamot", "nars", "nurse", "pulis", "police",
    "sundalo", "militar", "drayber", "tsuper", "negosyante", "manggagawa", "empleado",
    
    # Places
    "lugar", "pwesto", "bahay", "tahanan", "kwarto", "silid",
    "kusina", "banyo", "palikuran", "sala", "salas", "hardin", "bakuran", "garahe",
    "paaralan", "eskwela", "eskwelahan", "unibersidad", "kolehiyo",
    "silid-aralan", "classroom", "library", "aklatan",
    "opisina", "tanggapan", "pabrika", "tindahan", "tindaan", "store",
    "palengke", "merkado", "market", "mall", "shopping",
    "sinehan", "sine", "restawran", "kainan", "ospital",
    "simbahan", "church", "parke", "park",
    "kalsada", "daan", "kalye", "street", "eskinita", "tulay", "bridge",
    "istasyon", "station", "terminal",
    "bayan", "lungsod", "siyudad", "baryo", "nayon",
    "probinsya", "province", "rehiyon", "region", "bansa", "country", "mundo", "world",
    
    # Things
    "pera", "salapi", "kwarta", "trabaho", "hanapbuhay",
    "damit", "kasuotan", "pantalon", "pants", "sando", "shirt",
    "sapatos", "tsinelas", "medyas", "socks", "jacket", "dyaket",
    "libro", "aklat", "kuwaderno", "notebook", "papel",
    "lapis", "pensil", "ballpen", "bolpen", "bag",
    "telepono", "cellphone", "phone", "kompyuter", "computer",
    "telebisyon", "radyo", "radio",
    "sasakyan", "kotse", "auto", "jeep", "jeepney", "bus", "tren", "train",
    "eroplano", "plane", "barko", "boat", "bisikleta", "bike", "motorsiklo", "motor",
    "mesa", "table", "silya", "upuan", "chair", "kama", "higaan",
    "aparador", "cabinet",
    "payong", "umbrella", "relo", "orasan", "watch", "salamin", "mirror",
    "gunting", "scissors", "kutsilyo", "knife", "kutsara", "spoon",
    "tinidor", "fork", "plato", "plate", "baso", "glass",
    
    # Food & Drink
    "pagkain", "inumin",
    "almusal", "breakfast", "tanghalian", "lunch", "hapunan", "dinner", "meryenda", "snack",
    "kanin", "rice", "bigas", "tinapay", "bread", "ulam", "viand",
    "karne", "meat", "manok", "chicken", "baboy", "pork", "baka", "beef",
    "isda", "fish", "hipon", "shrimp", "itlog", "egg",
    "gulay", "vegetable", "talong", "eggplant", "kamatis", "tomato",
    "sibuyas", "onion", "bawang", "garlic",
    "prutas", "fruit", "mansanas", "apple", "saging", "banana",
    "mangga", "mango", "orange", "dalandan", "ubas", "grapes",
    "tubig", "water", "kape", "coffee", "tsaa", "tea", "gatas", "milk",
    "juice", "katas", "softdrinks", "coke",
    "adobo", "sinigang", "lechon", "pancit", "lumpia", "sisig",
    "kare-kare", "bulalo", "tinola",
    
    # Time
    "oras", "time", "panahon",
    "ngayon", "now", "mamaya", "later", "mamayang", "bukas", "tomorrow",
    "kahapon", "yesterday", "kanina", "earlier", "kaninang",
    "umaga", "morning", "tanghali", "noon", "hapon", "afternoon",
    "gabi", "evening", "night", "hatinggabi", "midnight", "madaling-araw", "dawn",
    "araw", "day", "Lunes", "Monday", "Martes", "Tuesday",
    "Miyerkules", "Wednesday", "Huwebes", "Thursday", "Biyernes", "Friday",
    "Sabado", "Saturday", "Linggo", "Sunday",
    "linggo", "week", "buwan", "month", "taon", "year",
    "araw-araw", "everyday", "bawat", "minsan", "sometimes",
    "madalas", "often", "lagi", "always", "kadalasan", "usually",
    "hindi", "kailanman", "never",
    
    # Particles & Connectors
    "ang", "ng", "sa", "na", "ay",
    "at", "and", "o", "or", "pero", "but", "ngunit", "however",
    "kasi", "because", "dahil", "kaya", "so", "kung", "if",
    "kapag", "when", "habang", "while", "para", "for",
    "upang", "saka", "and then",
    "naman", "din", "rin", "lang", "lamang", "talaga", "really",
    "sobra", "very", "masyado", "too much",
    "pala", "ba", "po", "ho", "pa", "nga", "indeed",
    "daw", "raw", "they say", "muna", "first",
    "sana", "hopefully", "yata", "maybe", "siguro", "probably",
    
    # Questions
    "ano", "what", "sino", "who", "saan", "where",
    "kailan", "when", "bakit", "why", "paano", "how",
    "ilan", "how many", "magkano", "how much",
    "alin", "which", "kanino", "whose",
    
    # Expressions
    "oo", "yes", "hindi", "no", "opo", "yes po",
    "ewan", "I don't know", "alam",
    "sige", "okay", "ayos", "fine", "tama", "correct", "mali", "wrong",
    "gusto", "want", "like", "ayoko", "don't want", "ayaw", "refuse", "ibig", "desire",
    "pwede", "can", "puwede", "maaari", "may", "kailangan", "need", "dapat", "should",
    "kumusta", "how are you", "salamat", "thank you",
    "pasensya", "sorry", "patawad", "walang", "anuman", "you're welcome",
    "paalam", "goodbye",
    
    # Location
    "dito", "here", "diyan", "there", "doon", "far",
    "mula", "from", "hanggang", "until",
    "loob", "inside", "labas", "outside", "itaas", "above", "ibaba", "below",
    "tabi", "beside", "harap", "front", "likod", "back",
    "kanan", "right", "kaliwa", "left", "gitna", "middle",
    
    # Quantity
    "lahat", "all", "wala", "none", "may", "have",
    "marami", "many", "konti", "few", "kaunti", "little",
    "kulang", "lacking", "sapat", "enough",
    "higit", "more than", "lampas", "over",
    
    # Numbers
    "isa", "dalawa", "tatlo", "apat", "lima",
    "anim", "pito", "walo", "siyam", "sampu",
    "labing-isa", "labindalawa", "dalawampu", "tatlumpu",
    "daan", "libo", "milyon",
    
    # Body Parts
    "katawan", "body", "ulo", "head", "mukha", "face",
    "mata", "eyes", "ilong", "nose", "bibig", "mouth",
    "tainga", "ear", "ngipin", "teeth", "dila", "tongue",
    "buhok", "hair", "leeg", "neck", "balikat", "shoulder",
    "braso", "arm", "kamay", "hand", "daliri", "finger",
    "dibdib", "chest", "tiyan", "stomach", "puwit", "buttocks",
    "hita", "thigh", "tuhod", "knee", "binti", "leg",
    "paa", "foot", "puso", "heart",
    
    # Weather & Nature
    "ulan", "rain", "hangin", "wind", "bagyo", "typhoon",
    "lindol", "earthquake", "kidlat", "lightning", "kulog", "thunder",
    "ulap", "cloud", "bituin", "star",
    "tag-init", "summer", "tag-ulan", "rainy season",
    "dagat", "sea", "ilog", "river", "bundok", "mountain",
    "puno", "tree", "bulaklak", "flower", "damo", "grass", "hayop", "animal",
    
    # Modern Slang
    "petmalu", "cool", "lodi", "idol", "werpa", "power",
    "syota", "girlfriend", "boyfriend", "jowa", "partner",
    "beshie", "bestfriend", "besh", "mare", "friend", "pare",
    "bro", "brother", "sis", "sister",
    "chika", "talk", "chismis", "gossip", "eme", "nonsense",
]

# Communication corpus for better predictions (500+ common phrases)
COMMUNICATION_CORPUS = [
    # Greetings
    "kumusta ka na", "kumusta ka", "mabuti naman", "ayos lang", "okay lang naman",
    "hello kumusta", "hi kumusta ka", "hey ano na", "oy kumusta",
    
    # Common Questions
    "ano ginagawa mo", "saan ka pupunta", "saan ka na", "kailan ka darating",
    "bakit ka nandito", "paano pumunta doon", "magkano yan", "sino kasama mo",
    "ano gusto mo", "ano kain mo", "saan ka nakatira", "anong oras na",
    
    # Yes/No
    "oo naman", "oo sige", "oo tara", "hindi naman", "hindi pa",
    "hindi ko alam", "ewan ko", "siguro nga", "baka oo",
    
    # Food
    "kumain ka na ba", "gutom na ako", "tara kain tayo", "ano ulam natin",
    "sama ka kumain", "busog na ako", "masarap yan", "ayoko nyan",
    "gusto ko ng", "bili tayo pagkain", "order tayo", "may tubig ka",
    
    # Going Places
    "tara na", "sama ka", "saan tayo pupunta", "dito lang ako",
    "andito na ako", "pauwi na ako", "papunta na ako", "nasaan ka na",
    "malapit na ako", "traffic pa", "otw na ako", "wait lang",
    
    # Time
    "mamaya na lang", "bukas na lang", "next time na", "maya maya",
    "konti lang", "saglit lang", "ngayon na", "kanina pa ako",
    "tagal mo naman", "bilisan mo", "dali na",
    
    # Feelings
    "pagod na ako", "antok na ako", "masaya ako", "okay lang ako",
    "sad ako ngayon", "stressed ako", "excited ako", "miss na kita",
    
    # School/Work
    "may pasok ba", "wala pasok", "may klase pa", "may exam bukas",
    "may trabaho pa ako", "off ko ngayon", "bakasyon na", "late na naman",
    
    # Plans
    "sama ka sakin", "tara labas tayo", "gala tayo", "manood tayo",
    "game ka ba", "free ka ba", "pwede ka ba", "ano plans mo",
    "sige game", "di ako pwede", "may lakad pa ako",
    
    # Money
    "magkano to", "mahal naman", "mura lang yan", "bili mo na",
    "wala akong pera", "may pera ka ba", "bayad ko na", "sweldo na bukas",
    
    # Family & Friends
    "nandito si mama", "nasaan si papa", "kasama ko kapatid ko",
    "sabi niya", "ayaw niya", "gusto niya", "alam niya ba",
    
    # Tech & Communication
    "chat mo ako", "text mo ako", "call kita later", "message mo ako",
    "reply ka naman", "seen mo lang", "may wifi ba", "lowbat na ako",
    
    # Weather
    "init ngayon", "mainit na naman", "umuulan ba", "umulan kanina",
    "malamig ngayon", "ganda ng panahon", "maraming tao",
    
    # Agreement
    "tama ka", "mali ka", "oo nga", "hindi nga", "talaga ba",
    "totoo yan", "joke lang", "seryoso ako", "sure ka ba",
    
    # Help & Thanks
    "tulungan mo ako", "help naman", "pakiusap", "please lang",
    "thank you", "salamat talaga", "salamat ha", "walang anuman",
    
    # Apologies
    "sorry na", "pasensya na", "di ko sinasadya", "okay lang", "ayos lang yun",
    
    # Taglish
    "bye na ako", "see you bukas", "good morning sa lahat", "good night na",
    "thank you very much", "excited na ako", "ang ganda naman", "so cute",
    "grabe naman", "super sarap", "sobrang init",
    
    # Social Media
    "hahaha funny mo", "lol true", "omg talaga ba", "grabe ka",
    
    # Common Starters
    "alam mo ba", "alam mo na ba", "sabi ko na nga ba", "gusto ko ng",
    "pwede ba", "kailangan ko ng", "hindi ko alam",
    
    # Activities
    "ligo muna ako", "tulog na ako", "gising na ako", "aalis na ako",
    "uuwi na ako", "pauwi na",
    
    # Descriptions
    "maganda yan", "ang ganda", "ang pangit naman", "ang laki",
    "masarap talaga", "matamis masyado",
    
    # With 'Na'
    "kumain ka na ba", "tapos ka na ba", "alis na ako", "kain na tayo",
    "tara na", "sige na", "wala na",
    
    # Negations
    "wala akong alam", "ayoko nyan", "ayaw ko", "wag na", "hindi pa",
    
    # With 'Ba'
    "okay ka ba", "kaya mo ba", "nandito ka ba", "pupunta ka ba",
    "totoo ba yan",
    
    # With 'Lang'
    "joke lang", "konti lang", "sandali lang", "dito lang", "okay lang",
    
    # Natural Phrases
    "ano ulam niyo", "nasaan yung", "kanino yan", "para saan yan",
    "talaga ba yan", "grabe ka naman", "hay nako",
    
    # More Taglish
    "see you na lang", "call me pag free ka", "text mo lang ako",
    "chat na lang tayo", "send mo sakin",
]

# Built-in shortcuts (text speak)
FILIPINO_SHORTCUTS = {
    # Vowel Removal
    "lng": "lang", "nmn": "naman", "ksi": "kasi", "kng": "kung",
    "tlg": "talaga", "sna": "sana", "dpt": "dapat",
    "mgnd": "maganda", "mgndng": "magandang",
    
    # Number Substitution
    "d2": "dito", "dn": "doon", "dyn": "diyan",
    
    # Phonetic
    "pde": "pwede", "pwd": "pwede", "pede": "pwede",
    "kya": "kaya", "dko": "hindi ko",
    
    # Pronouns
    "aq": "ako", "aqo": "ako", "ikw": "ikaw",
    "xa": "siya", "sya": "siya", "cya": "siya",
    "kmi": "kami", "tyo": "tayo", "kyo": "kayo", "cla": "sila",
    
    # Common
    "nd": "hindi", "nde": "hindi", "hnd": "hindi", "hdi": "hindi",
    "nu": "ano", "sn": "sino", "san": "saan",
    "bkt": "bakit", "bat": "bakit",
    "pno": "paano", "panu": "paano",
    "klan": "kailan", "kln": "kailan",
    
    # Actions
    "kn": "kumain", "pnta": "pumunta",
    
    # Expressions
    "sge": "sige", "gsto": "gusto", "ayko": "ayoko",
    
    # Time
    "ngyn": "ngayon", "ngaun": "ngayon",
    "mya": "mamaya", "mmya": "mamaya",
    "knina": "kanina", "knna": "kanina",
    "bkas": "bukas", "khpon": "kahapon",
    
    # Greetings
    "kumsta": "kumusta", "musta": "kumusta",
    
    # Social
    "brb": "be right back", "omg": "oh my god",
    "btw": "by the way", "idk": "i don't know",
    "tbh": "to be honest", "lol": "laughing",
    "jk": "just kidding",
    
    # Others
    "asap": "as soon as possible", "otw": "on the way",
    "omw": "on my way",
}

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
                dl_score * 0.20 +      # 20% edit distance (NEW!)
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
            completions = ngram_model.get_completion_suggestions(token, context, max_results=8)
            
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
        predictions = ngram_model.get_next_word_suggestions(context, max_results=6)
        
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
