import litserve as lit
from agent_core import StockDataRetrieverTool, FinancialAgentBuilder

# Query example
# query = """I have 10k dollars Now analyze APPL stock and MSFT stock to let me know where to invest
#             this money and how many stock will I get it, Give me the last 3 months historical close prices of
#             APPL and MSFT. Respond with a comparative summary on closing prices and recommended stock to invest."""


class IntrospectionAgentAPI(lit.LitAPI):
    def __init__(self):
        # Instantiate tool and agent
        self.stock_data_tool = StockDataRetrieverTool()
        self.financial_agent_builder = None

    def setup(self, device):
        self.financial_agent_builder = FinancialAgentBuilder(data_tool=self.stock_data_tool)

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, x, **kwargs):
        introspective_agent = self.financial_agent_builder.create_introspective_agent(verbose=True)
        response = introspective_agent.chat(x)
        return response

    def encode_response(self, output, **kwargs):
        return {"agent": output}


if __name__ == "__main__":
    api = IntrospectionAgentAPI()
    # pass this parameter "spec=lit.OpenAISpec()", if you want to use the API as
    # http://localhost:8002/v1/chat/completion
    # else http://localhost:8002/predict
    server = lit.LitServer(api, accelerator="auto", spec=lit.OpenAISpec())
    server.run(port=8002)
