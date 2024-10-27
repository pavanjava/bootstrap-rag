retrieval_grader_template = """You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains any information or keywords related to the user 
    question,grade it as relevant. This is a very lenient test - the document does not need to fully answer the question 
    to be considered relevant. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
    Also provide a brief explanation for your decision.
                    
    Return your response as a JSON with two keys: 'score' (either 'yes' or 'no') and 'explanation'.
                     
    Here is the retrieved document: 
    {document}
                    
    Here is the user question: 
    {question}
    """

hallucination_grading_template = """You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation.
    
    Here are the facts:
    {documents} 

    Here is the answer: 
    {generation}
    """

answer_generating_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Context: {context} 
    Answer: 
    """

answer_grading_template = """You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     
    Here is the answer:
    {generation} 

    Here is the question: {question}
    """
