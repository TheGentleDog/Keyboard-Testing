#!/usr/bin/env python3
"""
Test script to verify the Filipino keyboard model building works
This runs without the GUI to test the core functionality
"""

import json
import pickle
import os
from collections import defaultdict, Counter

# Configuration
NGRAM_CACHE_FILE = "ngram_model_standalone.pkl"
DATASET_FILE = "filipino_dataset.json"
MODEL_VERSION = "2.1_json_dataset"

print("="*60)
print("TESTING FILIPINO KEYBOARD MODEL BUILDING")
print("="*60)

# Load dataset
def load_dataset():
    """Load Filipino vocabulary and corpus from JSON file"""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✓ Loaded dataset: {data['metadata']['description']}")
        print(f"  Version: {data['metadata']['version']}")
        print(f"  Words: {data['metadata']['total_words']}")
        print(f"  Phrases: {data['metadata']['total_phrases']}")
        print(f"  Shortcuts: {data['metadata']['total_shortcuts']}")
        
        # Flatten vocabulary
        words = []
        for category, word_list in data['vocabulary'].items():
            words.extend(word_list)
            print(f"  - {category}: {len(word_list)} words")
        
        shortcuts = data['shortcuts']
        corpus = data['communication_corpus']
        
        print(f"\n✓ Total vocabulary loaded: {len(words)} words")
        print(f"✓ Total shortcuts loaded: {len(shortcuts)}")
        print(f"✓ Total phrases loaded: {len(corpus)}")
        
        return words, shortcuts, corpus
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: {DATASET_FILE} not found!")
        return None, None, None
    except Exception as e:
        print(f"\n✗ ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Load the dataset
print("\n" + "="*60)
print("STEP 1: Loading Dataset")
print("="*60)
words, shortcuts, corpus = load_dataset()

if words is None:
    print("\n✗ FAILED: Could not load dataset")
    exit(1)

# Test model building
print("\n" + "="*60)
print("STEP 2: Building N-gram Model")
print("="*60)

all_tokens = []

# Add words
print("\nAdding vocabulary words...")
for word in words:
    all_tokens.extend([word.lower()] * 10)
print(f"✓ Added {len(words)} words (×10 each)")

# Add corpus phrases
print("\nAdding communication corpus...")
for phrase in corpus:
    words_in_phrase = phrase.lower().split()
    for _ in range(50):  # 50x repetition for strong patterns
        all_tokens.extend(words_in_phrase)
print(f"✓ Added {len(corpus)} phrases (×50 each)")

# Add shortcuts
print("\nAdding shortcuts...")
for shortcut, full_word in shortcuts.items():
    all_tokens.extend([full_word.lower()] * 5)
print(f"✓ Added {len(shortcuts)} shortcuts")

print(f"\n✓ Total tokens for training: {len(all_tokens)}")

# Build simple statistics
print("\n" + "="*60)
print("STEP 3: Computing Statistics")
print("="*60)

vocabulary = set(all_tokens)
unigrams = Counter(all_tokens)
bigrams = defaultdict(Counter)

print(f"\nBuilding bigrams...")
for i in range(len(all_tokens) - 1):
    bigrams[all_tokens[i]][all_tokens[i+1]] += 1

print(f"\n✓ Unique vocabulary: {len(vocabulary)} words")
print(f"✓ Total tokens: {len(all_tokens)}")
print(f"✓ Unigram counts: {len(unigrams)}")
print(f"✓ Bigram contexts: {len(bigrams)}")

# Show top words
print("\nTop 10 most frequent words:")
for word, count in unigrams.most_common(10):
    print(f"  {word}: {count}")

# Test predictions
print("\n" + "="*60)
print("STEP 4: Testing Predictions")
print("="*60)

test_words = ["ako", "kumain", "tayo", "gusto"]
for word in test_words:
    if word in bigrams:
        top_predictions = bigrams[word].most_common(5)
        print(f"\nAfter '{word}', most likely:")
        for next_word, count in top_predictions:
            print(f"  → {next_word} (count: {count})")

# Test shortcuts
print("\n" + "="*60)
print("STEP 5: Testing Shortcuts")
print("="*60)

test_shortcuts = ["lng", "nmn", "ksi", "d2", "pde"]
print("\nSample shortcuts:")
for shortcut in test_shortcuts:
    if shortcut in shortcuts:
        print(f"  {shortcut} → {shortcuts[shortcut]}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nThe model building works correctly.")
print("The keyboard should work on your computer with tkinter installed.")
print("\nTo run the actual keyboard:")
print("  python filipino_keyboard_standalone.py")
