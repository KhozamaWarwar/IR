from typing import List
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker


class PreprocessingService:
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    def __init__(self, text):
        self.text = text

    def preprocess(self):
        tokens = word_tokenize(self.text.lower())  # Tokenization and lowercasing
        tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric tokens
        tokens = [token for token in tokens if token not in self.stop_words]  # Remove stop words
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
        tokens = [self.stemmer.stem(token) for token in tokens]  # Stemmatization
        return " ".join(tokens)

    def correct_sentence_spelling(self, tokens: List[str]):
        spell = SpellChecker()
        misspelled = spell.unknown(tokens)
        for i, token in enumerate(tokens):
            if token in misspelled:
                corrected = spell.correction(token)
                if corrected is not None:
                    tokens[i] = corrected
        return " ".join(tokens)
