import logging
import os

from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from mem0 import Memory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TravelAgentAI:
    def __init__(self, config: Dict):
        self.config = config

        # Initialize LLM and Memory
        logger.info("Initializing LLM and Memory...")
        self.llm = ChatOllama(model=config["llm"]["config"]["model"])
        self.memory = Memory.from_config(config_dict=config)
        logger.info("LLM and Memory initialized successfully.")

        # Define the prompt
        logger.info("Defining prompt...")
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful travel agent AI. Use the provided context to personalize your responses and remember user preferences and past interactions. 
            Provide travel recommendations, itinerary suggestions, and answer questions about destinations. 
            If you don't have specific information, you can make general suggestions based on common travel knowledge."""),
            MessagesPlaceholder(variable_name="context"),
            HumanMessage(content="{input}")
        ])
        logger.info("Prompt defined successfully.")

    def retrieve_context(self, query: str, user_id: str) -> List[Dict]:
        """Retrieve relevant context from Memory."""
        logger.info(f"Retrieving context for query: {query}, user_id: {user_id}")
        memories = self.memory.search(query, user_id=user_id)
        serialized_memories = ' '.join([mem["memory"] for mem in memories])
        logger.info(f"Serialized memories: {serialized_memories}")
        context = [
            {
                "role": "system",
                "content": f"Relevant information: {serialized_memories}"
            },
            {
                "role": "user",
                "content": query
            }
        ]
        logger.info(f"Context retrieved: {context}")
        return context

    def generate_response(self, input: str, context: List[Dict]) -> str:
        """Generate a response using the language model."""
        logger.info(f"Generating response for input: {input} with context: {context}")
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "input": input
        })
        logger.info(f"Generated response: {response.content}")
        return response.content

    def save_interaction(self, user_id: str, user_input: str, assistant_response: str):
        """Save the interaction to Memory."""
        logger.info(f"Saving interaction for user_id: {user_id}")
        interaction = [
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]
        self.memory.add(interaction, user_id=user_id)
        logger.info(f"Interaction saved: {interaction}")

    def chat_turn(self, user_input: str, user_id: str) -> str:
        """Handle a single turn of chat."""
        logger.info(f"Starting chat turn for user_id: {user_id} with input: {user_input}")

        # Retrieve context
        context = self.retrieve_context(user_input, user_id)

        # Generate response
        response = self.generate_response(user_input, context)

        # Save interaction
        self.save_interaction(user_id, user_input, response)

        logger.info(f"Chat turn completed. Response: {response}")
        return response


if __name__ == "__main__":
    logger.info("Welcome to your personal Travel Agent Planner! How can I assist you with your travel plans today?")
    user_id = os.environ.get("USER_ID")

    # Load the configuration
    logger.info("Loading configuration...")
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "langchain-agent-with-memory",
                "url": "http://localhost:6333",
                "api_key": "th3s3cr3tk3y",
                "embedding_model_dims": 768,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.2:latest",
                "temperature": 0,
                "max_tokens": 8000,
                "ollama_base_url": "http://localhost:11434",
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": "http://localhost:11434",
            },
        },
    }
    logger.info("Configuration loaded successfully.")

    # Instantiate the TravelAgentAI class
    logger.info("Instantiating TravelAgentAI...")
    travel_agent_ai = TravelAgentAI(config)
    logger.info("TravelAgentAI instantiated successfully.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            logger.info("Travel Agent: Thank you for using our travel planning service. Have a great trip!")
            break

        logger.info(f"Processing user input: {user_input}")
        response = travel_agent_ai.chat_turn(user_input, user_id)
        logger.info(f"Travel Agent: {response}")
