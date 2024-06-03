from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib


class DataRepresentationService:
    data_list = []
    tfidf_vectorizer = TfidfVectorizer()
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

    def __init__(self, data_list):
        self.data_list = data_list

    def create_folder(path):
        try:
            os.makedirs(path)
            print(f"Directory '{path}' created successfully")
        except FileExistsError:
            print(f"Directory '{path}' already exists")

    def represent_data(self, dataset_name):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data_list)

        self.create_folder(f'{self.desktop}/IR/{dataset_name.lower()}')
        joblib.dump(self.tfidf_vectorizer, f'{self.desktop}/IR/{dataset_name.lower()}/vectorizer.pkl', compress=True)
        joblib.dump(tfidf_matrix, f'{self.desktop}/IR/{dataset_name.lower()}/matrix.pkl', compress=True)