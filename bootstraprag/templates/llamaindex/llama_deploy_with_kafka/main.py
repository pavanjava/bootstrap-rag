from llama_deploy import LlamaDeployClient, ControlPlaneConfig

# points to deployed control plane
client = LlamaDeployClient(ControlPlaneConfig())

query = "what is attention?"
session = client.create_session()

result1 = session.run(service_name="rag_workflow_with_retry_query_engine", query=query)
result2 = session.run(service_name="rag_workflow_with_retry_source_query_engine", query=query)
result3 = session.run(service_name="rag_workflow_with_retry_guideline_query_engine", query=query)

print(f'response from rag_workflow_with_retry_query_engine is {result1}')
print(f'response from rag_workflow_with_retry_source_query_engine is {result2}')
print(f'response from rag_workflow_with_retry_guideline_query_engine is {result3}')
