# =============================================================================
# config.py — Global configuration and constants
# =============================================================================

MAX_SUGGESTIONS   = 5
NGRAM_CACHE_FILE  = "ngram_model_standalone.pkl"
USER_LEARNING_FILE = "user_learning.json"
DATASET_FILE      = "filipino_dataset.json"
MODEL_VERSION     = "2.1_json_dataset"

# Dwell engine settings
DWELL_POLL_MS  = 50     # how often we update (ms)
DWELL_ENABLED  = True   # can be toggled from Settings
DWELL_MIN_MS   = 600    # ms of continuous hover required to fire
