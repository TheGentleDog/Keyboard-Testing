import msvcrt
import os

SHORTCUT_DICT = {
    "kng": ["kung"],
    "kc": ["kasi"],
    "ksi": ["kasi"],
    "mron": ["meron"],
    "mrn": ["meron"],
    "d2": ["dito"],
    "dn": ["doon"],
    "jan": ["diyan"],
    "san": ["saan"],
    "tlg": ["talaga"],
    "pde": ["pwede"],
    "pd": ["pwede"],
    "aq": ["ako"],
    "xa": ["siya"],
    "sya": ["siya"],
    "u": ["ikaw", "oo"],
    "lng": ["lang"],
    "nmn": ["naman"],
    "sna": ["sana"]
}

NGRAM_SIZE = 2
MAX_SUGGESTIONS = 5


# ---------------- Utility functions ----------------
def clear_screen():
    os.system("cls")


def get_current_token(text: str) -> str:
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]


def char_ngrams(word, n=2):
    return {word[i:i+n] for i in range(len(word) - n + 1)} if len(word) >= n else {word}


def ngram_similarity(a, b, n=2):
    na = char_ngrams(a, n)
    nb = char_ngrams(b, n)
    if not na or not nb:
        return 0
    return len(na & nb) / len(na | nb)


def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insert = previous_row[j + 1] + 1
            delete = current_row[j] + 1
            replace = previous_row[j] + (ca != cb)
            current_row.append(min(insert, delete, replace))
        previous_row = current_row
    return previous_row[-1]


def realtime_suggestions(token, shortcut_dict):
    if not token:
        return []

    candidates = []
    for shortcut, words in shortcut_dict.items():
        sim = ngram_similarity(token, shortcut, NGRAM_SIZE)
        if sim > 0:
            for word in words:
                dist = levenshtein(token, shortcut)
                candidates.append((word, sim, dist))

    candidates.sort(key=lambda x: (-x[1], x[2]))

    seen = set()
    results = []
    for word, _, _ in candidates:
        if word not in seen:
            results.append(word)
            seen.add(word)
        if len(results) >= MAX_SUGGESTIONS:
            break

    return results


def replace_current_token(text, replacement):
    tokens = text.split()
    if tokens:
        tokens[-1] = replacement
    return " ".join(tokens) + " "


# ---------------- Main IME loop ----------------
def main():
    text = ""
    suggestion_index = 0
    current_suggestions = []

    print("Filipino Shortcut Keyboard IME (N-gram + Levenshtein)")
    print("--------------------------------------------------")
    print("ESC   : exit")
    print("BACK  : backspace")
    print("ENTER : space")
    print("TAB   : accept highlighted suggestion")
    print("UP/DOWN : navigate suggestions")
    print("--------------------------------------------------\n")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()

            if key == b'\x1b':  # ESC
                break

            elif key == b'\x08':  # BACKSPACE
                text = text[:-1]

            elif key == b'\r':  # ENTER / space
                text += " "

            elif key == b'\t':  # TAB = accept highlighted suggestion
                if current_suggestions:
                    text = replace_current_token(text, current_suggestions[suggestion_index])
                    suggestion_index = 0

            elif key in (b'\x00', b'\xe0'):  # special keys (arrows)
                special = msvcrt.getch()
                if special == b'H':  # UP arrow
                    if current_suggestions:
                        suggestion_index = (suggestion_index - 1) % len(current_suggestions)
                elif special == b'P':  # DOWN arrow
                    if current_suggestions:
                        suggestion_index = (suggestion_index + 1) % len(current_suggestions)

            else:
                try:
                    text += key.decode("utf-8")
                except UnicodeDecodeError:
                    pass

            # Update suggestions
            token = get_current_token(text)
            current_suggestions = realtime_suggestions(token, SHORTCUT_DICT)
            suggestion_index = 0 if not current_suggestions else suggestion_index % len(current_suggestions)

            # Display
            clear_screen()
            print("Filipino Shortcut Keyboard IME (N-gram + Levenshtein)")
            print("--------------------------------------------------")
            print("Typed text :", text)
            print("Token      :", token)
            print("Suggestions:")
            for i, s in enumerate(current_suggestions):
                prefix = "-> " if i == suggestion_index else "   "
                print(f"{prefix}{s}")
            print("--------------------------------------------------")

    print("\nFinal Output:")
    print(text)


if __name__ == "__main__":
    main()
