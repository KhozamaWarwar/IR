from sklearn.metrics.pairwise import cosine_similarity
import ir_datasets


class MatchingAndRankingService:

    def __init__(self, dataset_id):
        self.dataset = ir_datasets.load(dataset_id)

    def match_and_rank(self, query_vector, tfidf):
        similarity_scores = cosine_similarity(query_vector, tfidf)
        relevant_documents_indices = [b[0] for b in sorted(enumerate(similarity_scores[0]), key=lambda i: i[1], reverse=True)][:10]

        documents = []
        for relevant_document_index in relevant_documents_indices:
            relevant_document = self.dataset.docs_iter()[relevant_document_index]
            # documents.append({
            #     "doc_id": relevant_document.doc_id,
            #     "title": relevant_document.title.replace("\n      ", " ").replace("\n     ", " ").strip(),
            #     "condition": relevant_document.condition.replace("\n      ", " ").replace("\n     ", " ").strip(),
            #     "summary": relevant_document.summary.replace("\n      ", " ").replace("\n     ", " ").strip(),
            #     "detailed_description": relevant_document.detailed_description.replace("\n      ", " ").replace("\n     ", " ").strip(),
            #     "eligibility": relevant_document.eligibility.replace("\n      ", " ").replace("\n     ", " ").strip()
            # })
            documents.append(relevant_document.doc_id)
        return documents
