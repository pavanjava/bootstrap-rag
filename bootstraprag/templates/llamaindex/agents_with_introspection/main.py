from agent_core import StockDataRetrieverTool, FinancialAgentBuilder

# Instantiate tool and agent
stock_data_tool = StockDataRetrieverTool()
financial_agent_builder = FinancialAgentBuilder(data_tool=stock_data_tool)
introspective_agent = financial_agent_builder.create_introspective_agent(verbose=True)

# Query example
query = """I have 10k dollars Now analyze APPL stock and MSFT stock to let me know where to invest 
            this money and how many stock will I get it, Give me the last 3 months historical close prices of APPL and MSFT. 
            Respond with a comparative summary on closing prices and recommended stock to invest."""
response = introspective_agent.chat(query)
print(response)