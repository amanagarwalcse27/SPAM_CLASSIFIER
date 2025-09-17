import nltk
import os
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Set NLTK data path for Streamlit Cloud
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

ps = PorterStemmer()

def to_lower(text):
    return text.lower()

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_non_alphnum(tokens):
    return [i for i in tokens if i.isalnum()]

def removes_stop_punc(tokens):
    return [i for i in tokens if i not in stopwords.words('english') and i not in string.punctuation]

def stem_words(tokens):
    return [ps.stem(i) for i in tokens]

def full_pipeline(text):
    text = to_lower(text)
    tokens = tokenize(text)
    tokens = remove_non_alphnum(tokens)
    tokens = removes_stop_punc(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)



