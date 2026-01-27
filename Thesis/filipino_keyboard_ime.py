# -------------------------------------------------
# Filipino Shortcut Keyboard IME (TRUE REAL-TIME)
# Windows-only using msvcrt
# -------------------------------------------------

import msvcrt
import os

# Shortcut → normalized words
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


def get_current_token(text: str) -> str:
    if not text or text.endswith(" "):
        return ""
    return text.split()[-1]


def realtime_suggestions(current_token: str, shortcut_dict):
    suggestions = set()

    if not current_token:
        return []

    for shortcut, words in shortcut_dict.items():
        if shortcut.startswith(current_token):
            for word in words:
                suggestions.add(word)

    return sorted(suggestions)


def replace_current_token(text: str, replacement: str) -> str:
    tokens = text.split()
    tokens[-1] = replacement
    return " ".join(tokens) + " "


def clear_screen():
    os.system("cls")


def main():
    text = ""

    print("Filipino Shortcut Keyboard IME (Real-Time)")
    print("-----------------------------------------")
    print("ESC   : exit")
    print("BACK  : backspace")
    print("TAB   : accept first suggestion")
    print("-----------------------------------------\n")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()

            # Exit (ESC)
            if key == b'\x1b':
                break

            # Backspace
            elif key == b'\x08':
                text = text[:-1]

            # Enter (new word)
            elif key == b'\r':
                text += " "

            # Accept first suggestion
            elif key == b'\t':
                token = get_current_token(text)
                suggestions = realtime_suggestions(token, SHORTCUT_DICT)
                if suggestions:
                    text = replace_current_token(text, suggestions[0])

            # Ignore special keys (arrows, etc.)
            elif key in (b'\x00', b'\xe0'):
                msvcrt.getch()

            # Normal characters
            else:
                try:
                    text += key.decode("utf-8")
                except UnicodeDecodeError:
                    pass

            token = get_current_token(text)
            suggestions = realtime_suggestions(token, SHORTCUT_DICT)

            clear_screen()
            print("Filipino Shortcut Keyboard IME (Real-Time)")
            print("-----------------------------------------")
            print("Typed text :", text)
            print("Token      :", token)
            print("Suggestions:", ", ".join(suggestions))
            print("-----------------------------------------")

    print("\nFinal Output:")
    print(text)


if __name__ == "__main__":
    main()
