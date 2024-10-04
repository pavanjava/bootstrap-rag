from measure_retrieval_quality import MeasureRetrievalQuality

measure_rq = MeasureRetrievalQuality(collection_name='arxiv-titles-instructorxl-embeddings',
                                     dataset_path='Qdrant/arxiv-titles-instructorxl-embeddings')

# before tuning
print(f"avg(precision@5) = {measure_rq.compute_avg_precision_at_k(k=5)}")

measure_rq.tune_hnsw_configs()

# after tuning
print(f"avg(precision@5) = {measure_rq.compute_avg_precision_at_k(k=5)}")
