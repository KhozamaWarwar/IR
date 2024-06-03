from fastapi import FastAPI
from Services.QueryProcessingService import QueryProcessingService
from Services.MatchingAndRankingService import MatchingAndRankingService
import joblib
import os
from Models.Query import Query
from Models.Dataset import Dataset

dataset_info = Dataset()
app = FastAPI()
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

@app.on_event("startup")
async def startup_event():
    # Load your models or perform other startup tasks
    dataset_info.set_dataset_name("clinicaltrials/2017/trec-pm-2017")
    dataset_info.set_loaded_tvec(joblib.load(f'{desktop}/IR/{dataset_info.get_dataset_name()}/vectorizer.pkl'))
    dataset_info.set_tfidf(joblib.load(f'{desktop}/IR/{dataset_info.get_dataset_name()}/matrix.pkl'))

@app.post("/query/")
async def query_chosen_dataset(query: Query):
    if dataset_info.dataset_name != query.dataset:
        dataset_info.set_dataset_name(query.dataset)
        dataset_info.set_loaded_tvec(joblib.load(f'{desktop}/IR/{query.dataset.lower()}/vectorizer.pkl'))
        dataset_info.set_tfidf(joblib.load(f'{desktop}/IR/{query.dataset.lower()}/matrix.pkl'))

    query_vector = QueryProcessingService(query.query).process(dataset_info.get_loaded_tvec())
    match_object = MatchingAndRankingService(dataset_info.dataset_name)
    documents = match_object.match_and_rank(query_vector, dataset_info.get_tfidf())
    return documents

# @app.get("/preprocess/")
# async def preprocess_chosen_dataset(query: str):
#     pass
