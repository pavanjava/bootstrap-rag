from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.agent.introspective import ToolInteractiveReflectionAgentWorker, IntrospectiveAgentWorker
from llama_index.core.agent import FunctionCallingAgentWorker  # Import OpenAIAgentWorker here
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from datetime import datetime
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class StockDataRetrieverTool(BaseToolSpec):
    """Tool for retrieving stock data"""

    spec_functions = ['fetch_historical_prices']

    def fetch_historical_prices(self, ticker: str, start_date: str,
                                end_date: str = datetime.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
        """
        Retrieves historical stock prices for a given ticker within a date range.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): Start date for data in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            pd.DataFrame: DataFrame with stock price data.
        """
        return yf.download(tickers=ticker, start=start_date, end=end_date, progress=True)


class FinancialAgentBuilder:
    """Builder for creating an introspective financial agent with reflection capabilities"""

    def __init__(self, data_tool: BaseToolSpec):
        self.data_tool = data_tool
        self.data_tool_list = data_tool.to_tool_list()

    def create_introspective_agent(self, verbose: bool = True, include_main_worker: bool = True):
        critique_worker = self._build_critique_worker(verbose)
        reflection_worker = self._build_reflection_worker(critique_worker, verbose)
        main_worker = self._build_main_worker() if include_main_worker else None

        introspective_worker = IntrospectiveAgentWorker.from_defaults(
            reflective_agent_worker=reflection_worker,
            main_agent_worker=main_worker,
            verbose=verbose
        )

        initial_message = [
            ChatMessage(
                content="You are a financial assistant that helps gather historical prices "
                        "and provides a summary of the closing price analysis.",
                role=MessageRole.SYSTEM,
            )
        ]
        return introspective_worker.as_agent(chat_history=initial_message, verbose=verbose)

    def _build_critique_worker(self, verbose: bool):
        return FunctionCallingAgentWorker.from_tools(
            tools=self.data_tool_list,
            llm=OpenAI(model="gpt-4o-mini"),
            verbose=verbose
        )

    def _build_reflection_worker(self, critique_worker, verbose: bool):
        def stop_critique_on_pass(critique_str: str):
            return "[APPROVE]" in critique_str

        return ToolInteractiveReflectionAgentWorker.from_defaults(
            critique_agent_worker=critique_worker,
            critique_template=(
                "Please review the retrieval of historical prices with the specified time intervals. "
                "Evaluate the price for accuracy, efficiency, and adherence to truth. "
                "If everything is correct, write '[APPROVE]'; otherwise, write '[REJECT]'.\n\n{input_str}"
            ),
            stopping_callable=stop_critique_on_pass,
            correction_llm=OpenAI(model='gpt-4o'),
            verbose=verbose
        )

    def _build_main_worker(self):
        return OpenAIAgentWorker.from_tools(
            tools=self.data_tool_list,
            llm=OpenAI("gpt-4-turbo"),
            verbose=True
        )

