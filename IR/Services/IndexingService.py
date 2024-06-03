from collections import defaultdict
import ir_datasets
from ir_datasets.formats import ArgsMeStance

from Services.PreprocessingService import PreprocessingService
from nltk.tokenize import word_tokenize
from nltk import FreqDist


class IndexingService:
    datasets = []
    inverted_index = defaultdict(list)

    def __init__(self, dataset_id):
        loaded_dataset = ir_datasets.load(dataset_id)
        self.datasets = loaded_dataset.docs_iter()[0:10]

    def serialize(self, obj):
        if isinstance(obj, list):
            return [self.serialize(item) for item in obj]
        elif isinstance(obj, ArgsMeStance):
            return {key: self.serialize(value) for key, value in obj.items()}
        else:
            serial = ""
            for element in obj:
                serial = serial + element + " "
            return serial

    def index_data(self):
        for row in self.datasets:
            serialized_row = self.serialize(row)
            preprocess = PreprocessingService(serialized_row)
            text = word_tokenize(preprocess.preprocess())
            frequency = FreqDist(text)
            most_common_strings = [word for word, freq in frequency.most_common(10)]
            for word in most_common_strings:
                self.inverted_index[word].append(row.doc_id)
        print(dict(self.inverted_index))


index = IndexingService("clinicaltrials/2017/trec-pm-2017")
index.index_data()

