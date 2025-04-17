# retrieve candidate ICEs using various heuristics
import logging
import json
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class CandidateRetriever:
    def __init__(self, top_k, embedding_model_name, train_data, test_data, processor, ice_template):
        logging.info('Initialize Retriever ...')
        self.top_k = top_k
        self.encoder = SentenceTransformer(embedding_model_name)
        self.embeddings = []

        # data specs
        self.processor = processor

        # initialize
        self.train_queries = []
        self.train_answers = []
        self.train_query_answer_pairs = []
        self.train_ids = []

        for data_id in train_data:
            self.train_ids.append(data_id)
            self.train_queries.append(self.processor.extract_query(train_data[data_id]))
            self.train_answers.append(self.processor.extract_answer(train_data[data_id]))
            self.train_query_answer_pairs.append(self.processor.build_ice(ice_template, train_data[data_id]))

        # BM25 Index
        logging.info('Building train indices ...')
        self.index_train_queries = BM25Okapi([nltk.word_tokenize(query) for query in self.train_queries])
        self.index_train_answers = BM25Okapi([nltk.word_tokenize(answer) for answer in self.train_answers])
        self.index_train_query_answer_pairs = BM25Okapi([nltk.word_tokenize(query_answer) for query_answer in self.train_query_answer_pairs])

        # Embedding Index
        logging.info('Building train embeddings ...')
        self.train_query_embeddings = self.encoder.encode(self.train_queries)
        self.train_answer_embeddings = self.encoder.encode(self.train_answers)
        self.train_query_answer_embeddings = self.encoder.encode(self.train_query_answer_pairs)

        self.test_queries = []
        self.test_answers = []
        self.test_query_answer_pairs = []
        self.test_ids = []
        for data_id in test_data:
            self.test_ids.append(data_id)
            self.test_queries.append(self.processor.extract_query(test_data[data_id]))
            self.test_answers.append(self.processor.extract_answer(test_data[data_id]))
            self.test_query_answer_pairs.append(self.processor.build_ice(ice_template, test_data[data_id]))

        logging.info('Building test embeddings')
        self.test_query_embeddings = self.encoder.encode(self.test_queries)
        self.test_answer_embeddings = self.encoder.encode(self.test_answers)
        self.test_query_answer_embeddings = self.encoder.encode(self.test_query_answer_pairs)

        self.results = {
            'semantic_similarity_query': {},
            'semantic_similarity_answer': {},
            'semantic_similarity_query_answer': {},
            'surface_similarity_query': {},
            'surface_similarity_answer': {},
            'surface_similarity_query_answer': {},
        }

    def retrieve_semantic_sim(self):
        # similarities [num_test, num_train]
        logging.info('Calculating embedding similarities ...')
        query_similarities = self.encoder.similarity(self.test_query_embeddings, self.train_query_embeddings).numpy()
        logging.info('Query done!')
        answer_similarities = self.encoder.similarity(self.test_answer_embeddings, self.test_answer_embeddings).numpy()
        logging.info('Answer done !')
        query_answer_similarities = self.encoder.similarity(self.test_query_answer_embeddings, self.test_query_answer_embeddings).numpy()
        logging.info('Query answer done!')

        sorted_idx_query = np.argsort(-query_similarities, axis=1)
        sorted_idx_answer = np.argsort(-answer_similarities, axis=1)
        sorted_idx_query_answer = np.argsort(-query_answer_similarities, axis=1)

        for i in range(len(self.test_ids)):
            self.results['semantic_similarity_query'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_query[i][:self.top_k]]
            self.results['semantic_similarity_answer'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_answer[i][:self.top_k]]
            self.results['semantic_similarity_query_answer'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_query_answer[i][:self.top_k]]

    def retrieve_surface_sim(self):
        logging.info('BM25 Retrieving ...')
        for i in tqdm(range(len(self.test_ids))):
            tokenized_query = nltk.word_tokenize(self.test_queries[i])
            tokenized_answer = nltk.word_tokenize(self.test_answers[i])
            tokenized_query_answer_pair = nltk.word_tokenize(self.test_query_answer_pairs[i])

            sorted_idx_query = np.argsort(-self.index_train_queries.get_scores(tokenized_query))
            sorted_idx_answer = np.argsort(-self.index_train_answers.get_scores(tokenized_answer))
            sorted_idx_query_answer = np.argsort(-self.index_train_query_answer_pairs.get_scores(tokenized_query_answer_pair))

            self.results['surface_similarity_query'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_query[:self.top_k]]
            self.results['surface_similarity_answer'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_answer[:self.top_k]]
            self.results['surface_similarity_query_answer'][self.test_ids[i]] = [self.train_ids[j] for j in sorted_idx_query_answer[:self.top_k]]

    # TODO: structure similarity, complexity
    def retrieve_structure_sim(self, src):
        # assert src in ['query', 'answer', 'query_answer']
        pass

    def save(self, output_file):
        logging.info('Saving candidates to {} ...'.format(output_file))
        with open(output_file, 'w') as fp:
            fp.write(json.dumps(self.results)+'\n')

