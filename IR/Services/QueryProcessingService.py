from Services.PreprocessingService import PreprocessingService
from nltk.tokenize import word_tokenize


class QueryProcessingService:

    def __init__(self, query):
        self.query = query
        self.preprocessor = PreprocessingService(query)

    def process(self, loaded_tvec):
        corrected = self.preprocessor.correct_sentence_spelling(word_tokenize(self.query))
        self.preprocessor.text = corrected
        processed_query = self.preprocessor.preprocess()
        return loaded_tvec.transform([processed_query])


# hello = QueryProcessingService("Lung cancer NTRK1 58-year-old female Depression, Hypertension, Diabetes")
# print(hello.process())
