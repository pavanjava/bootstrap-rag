from llama_deploy import LlamaDeployClient, ControlPlaneConfig

# points to deployed control plane
client = LlamaDeployClient(ControlPlaneConfig())

query = "what are the challenges of mlops?"
session = client.create_session()

result3 = session.run(service_name="rag_workflow_with_retry_guideline_query_engine", query=query)

print(f'response from rag_workflow_with_retry_guideline_query_engine is {result3}')
