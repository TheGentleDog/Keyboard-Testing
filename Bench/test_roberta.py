from transformers import pipeline

fill = pipeline("fill-mask", model="jcblaise/roberta-tagalog-base", top_k=5)

# Test which mask token works
try:
    results = fill("Gusto kong [MASK] ng pagkain.")
    print("✓ [MASK] works:", results)
except Exception as e:
    print("✗ [MASK] failed:", e)

try:
    results = fill("Gusto kong <mask> ng pagkain.")
    print("✓ <mask> works:", results)
except Exception as e:
    print("✗ <mask> failed:", e)