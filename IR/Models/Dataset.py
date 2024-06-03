class Dataset:
    dataset_name = ""
    loaded_tvec = []
    tfidf = []

    def get_substring_until_delimiter(self, s, delimiter):
        return s.split(delimiter)[0]

    def set_loaded_tvec(self, loaded_tvec):
        self.loaded_tvec = loaded_tvec

    def set_tfidf(self, tfidf):
        self.tfidf = tfidf

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.get_substring_until_delimiter(self.dataset_name, "/")

    def get_tfidf(self):
        return self.tfidf

    def get_loaded_tvec(self):
        return self.loaded_tvec
