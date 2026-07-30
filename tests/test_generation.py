from rag_core.generation.query_engine import answer_query

result = answer_query("Top 5 roles meta is looking for", top_n=20, mode="hybrid", use_reranker=True)
print(result["answer"])
#print(result["source_nodes"])