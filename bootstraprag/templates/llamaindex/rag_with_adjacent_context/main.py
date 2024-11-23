from adjacent_context_rag import PrevNextPostprocessorDemo
# Example Usage
if __name__ == "__main__":
    # Initialize the class with your OpenAI API key
    demo = PrevNextPostprocessorDemo(
        data_directory="data",
        chunk_size=256,
    )

    # Load documents, parse them into nodes, and build the index
    demo.load_documents()
    demo.parse_documents_to_nodes()
    demo.build_index()

    # Query with and without the postprocessor
    print("Query with Postprocessor:")
    print(demo.query_with_postprocessor(query="What are the challenges of MLOps?"))

    print("\nQuery without Postprocessor:")
    print(demo.query_without_postprocessor(query="What are the challenges of MLOps?"))