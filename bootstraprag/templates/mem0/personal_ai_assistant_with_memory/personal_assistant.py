from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv())

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI()


class PersonalAIAssistant:
    def __init__(self):
        """
        Initialize the PersonalAITutor with memory configuration and OpenAI client.
        """
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "personal_assistant_memory",
                    "url": "http://localhost:6333",
                    "api_key": "th3s3cr3tk3y"
                }
            },
        }
        self.memory = Memory.from_config(config)
        self.client = client
        self.app_id = "assistant-app"

    def ask(self, question, user_id=None):
        """
        Ask a question to the AI and store the relevant facts in memory

        :param question: The question to ask the AI.
        :param user_id: Optional user ID to associate with the memory.
        """
        # Start a streaming chat completion request to the AI
        stream = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            stream=True,
            messages=[
                {"role": "system", "content": "You are a personal AI Assistant."},
                {"role": "user", "content": question}
            ]
        )
        # Store the question in memory
        self.memory.add(question, user_id=user_id, metadata={"app_id": self.app_id})
        response: str = ''
        # Print the response from the AI in real-time
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                response = response + chunk.choices[0].delta.content

        return response

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given user ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)


# Instantiate the PersonalAITutor
# ai_tutor = PersonalAIAssistant()

# Define a user ID
# user_id = "pavan_mantha"

# Ask a question
# ai_tutor.ask("What is my first question.", user_id=user_id)
