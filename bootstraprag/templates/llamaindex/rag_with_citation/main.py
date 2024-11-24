from rag_with_citation import CitationQueryEngineRAG

# Example usage
if __name__ == "__main__":
    engine = CitationQueryEngineRAG()
    response = engine.query("What are the benefits of MLOps?")
    print(response)