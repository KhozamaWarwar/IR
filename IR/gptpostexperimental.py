import ir_datasets
from Preprocess import Preprocess
from Services.DataRepresentationService import DataRepresentationService


def get_substring_until_delimiter(s, delimiter):
    return s.split(delimiter)[0]


def preprocess_represent_index(dataset_id):
    dataset_name = get_substring_until_delimiter(dataset_id, "/")
    loaded_dataset = ir_datasets.load(dataset_id)
    datasets = loaded_dataset.docs_iter()[0:100]

    preprocessor = Preprocess("")

    i = 0
    corpus = []
    for dataset in datasets:
        i += 1

        docstring = dataset[1] + " " + dataset[3] + " " + dataset[4] + " " + dataset[5]
        preprocessor.text = docstring
        corpus.append(preprocessor.preprocess())

        print("Preprocessing, feature extraction, and indexing completed successfully for document " + str(i) + ".")

    data_representation = DataRepresentationService(corpus)
    data_representation.represent_data(dataset_name)
