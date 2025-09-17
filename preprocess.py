import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load your saved model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# -----------------------------
# Text preprocessing functions
# -----------------------------

def to_lower(text):
    return text.lower()

def tokenize(text):
    # Simple regex-based tokenizer, no NLTK needed
    text = to_lower(text)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def remove_non_alphnum(tokens):
    return [token for token in tokens if token.isalnum()]

# Simple stopwords list (replace if you have your own)
STOPWORDS = set([
    'the','a','an','and','or','is','it','to','for','in','of','on','at','by','this','that'
])

def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

# Optional: simple stemming (can be skipped or implement your own)
def stem_words(tokens):
    # Very basic: remove common suffixes
    suffixes = ['ing', 'ly', 'ed', 's', 'es']
    stemmed = []
    for token in tokens:
        for suf in suffixes:
            if token.endswith(suf) and len(token) > len(suf)+2:
                token = token[:-len(suf)]
        stemmed.append(token)
    return stemmed

# Full preprocessing pipeline
def full_pipeline(text):
    tokens = tokenize(text)
    tokens = remove_non_alphnum(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)





